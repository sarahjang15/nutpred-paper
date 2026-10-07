import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from nutpred.preprocess import is_first_mapped
from sklearn.dummy import DummyClassifier
from sklearn.metrics import r2_score, mean_squared_error
import seaborn as sns
import nutpred.constants as C

logger = logging.getLogger(__name__)


MODEL_TAGS = ("xgboost", "random_forest", "hist_gradient_boosting", "support_vector",
              "linear", "knn", "mlp")


def _parse_fullnut_pred_col(pred_col: str):
    if pred_col.endswith(("_zero_prob", "_zero_pred")):
        return None, None, None
    for tag in MODEL_TAGS:
        needle = f"_{tag}_full_"
        if needle in pred_col:
            nutrient, feature_set = pred_col.split(needle, 1)
            return nutrient, tag, feature_set
    return None, None, None

def compare_feature_sets(metrics_df: pd.DataFrame = None, metrics_file_path: str = None, outdir: str = None):
    """Create a single heatmap showing model types (rows) × metrics (columns)."""
    logger.info("Creating model type × metrics comparison heatmap")
    
    # Load metrics data
    if metrics_df is not None:
        df = metrics_df
    elif metrics_file_path is not None:
        df = pd.read_csv(metrics_file_path)
    else:
        # Default to the standard metrics file
        df = pd.read_csv("metrics_test.csv")
    
    if len(df) == 0:
        logger.warning("No metrics found for comparison")
        return
    
    # Create model type column that combines Model and FeatureSet
    df['ModelType'] = df.apply(lambda row: 
        f"{row['Model']}_{row['FeatureSet']}" if row['Model'] != 'ing_pred' 
        else 'ing_pred', axis=1)
    
    # Define metrics to include
    metrics_to_plot = ['R2', 'RMSE', 'SMAPE']
    

    # Create separate heatmaps for each nutrient: ModelType (rows) × Group (columns)
    for nutrient in C.TARGET_NUT3:
        if not (df['Nutrient'] == nutrient).any():
            continue
        n_metrics = len(metrics_to_plot)
        fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 6))
        if n_metrics == 1:
            axes = [axes]
            
        for i, metric in enumerate(metrics_to_plot):
            if metric not in df.columns:
                logger.warning(f"Metric {metric} not found in dataframe")
                continue
                
            # Create pivot table: ModelType (rows) × Group (columns)
            pivot = df.query(f"Nutrient == '{nutrient}'").pivot_table(
                values=metric, 
                index='ModelType', 
                columns='Group', 
                aggfunc='mean'
            )
            
            # Reorder rows to put optimization first, then feature sets
            model_order = [col for col in pivot.index]
            pivot = pivot.reindex(model_order)
            pivot = pivot[[x for x in C.GROUP_COLS if x in pivot.columns]]
            
            # Create heatmap
            sns.heatmap(pivot, annot=True, fmt='.3f', cmap='Blues', 
                    cbar_kws={'label': metric}, ax=axes[i])
            axes[i].set_title(f'{metric} by Model Type and Group for {nutrient}', fontsize=12, fontweight='bold')
            axes[i].set_xlabel('Group', fontsize=10)
            axes[i].set_ylabel('Model Type', fontsize=10)
            
            # Rotate x-axis labels for better readability
            axes[i].tick_params(axis='x', rotation=90)
            axes[i].tick_params(axis='y', rotation=0)
        
        plt.tight_layout()
        
        # Save plot
        output_path = f"{outdir}/model_comparison_{nutrient}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Model comparison heatmap saved to {output_path}")

def create_scatterplots(pred_file_path: str = None, outdir: str = None):
    """True vs predicted scatterplots for every model prediction column.

    Only TEST rows are plotted: train rows carry in-sample predictions, which
    would make the fit look better than it is.
    """
    logger.info("Creating scatterplots (test rows only)....")

    plots_dir = os.path.join(outdir, "scatterplots")
    os.makedirs(plots_dir, exist_ok=True)

    food_df = pd.read_csv(pred_file_path)
    if "SampleType" in food_df.columns:
        food_df = food_df[food_df["SampleType"] == "test"]
    else:
        logger.warning("No SampleType column; scatterplots include every row.")

    hue_col = "ingnut_top3_mapped"
    if hue_col not in food_df.columns:
        logger.warning("Column '%s' not found; scatterplots will be uncolored.", hue_col)
        hue_col = None

    for pred_col in food_df.columns:
        nutrient, model_tag, feature_set = _parse_fullnut_pred_col(pred_col)
        if nutrient is None or nutrient not in food_df.columns:
            continue
        fig = plt.figure(figsize=(10, 8))
        sns.scatterplot(x=food_df[nutrient], y=food_df[pred_col], alpha=0.5,
                        hue=food_df[hue_col] if hue_col else None)
        plt.title(f'Test set: {nutrient} - {model_tag} - {feature_set} (n={len(food_df)})')
        plt.xlabel('True')
        plt.ylabel('Predicted')
        plt.axline((0, 0), (1, 1), color='black', linestyle='--')
        filename = f'scatterplot_{nutrient}_{model_tag}_full_{feature_set}_{hue_col or "all"}.png'
        fig.savefig(os.path.join(plots_dir, filename))
        plt.close(fig)


def _feature_group(col: str) -> str:
    """Coarse feature-group label used for the SHAP group-importance bar chart."""
    if col in C.NUT8:
        return "nut8"
    if col.startswith("binary_"):
        return "ingredient_binary"
    if col.startswith("score_"):
        return "ingredient_score"
    if col.startswith("umap_"):
        return "umap"
    if col.endswith(("_ing_pred", "_pred_ingnut_aux")):
        return "knowledge_guided"
    if "fortifier" in col.lower():
        return "fortifier"
    if "cluster" in col.lower():
        return "cluster"
    return "other"


def _predict_proba_positive(model, X: np.ndarray) -> np.ndarray:
    proba = model.predict_proba(X)
    return proba[:, 1] if proba.ndim == 2 and proba.shape[1] > 1 else proba.ravel()


def _shap_values_for(model_type: str, model, stage: str, X_background: np.ndarray, X_target: np.ndarray):
    """Return a (n_samples, n_features) array of SHAP values for one fitted
    regressor/classifier. Tree-based families (xgboost/random_forest/
    hist_gradient_boosting) get the fast, exact shap.TreeExplainer; every
    other family (linear/support_vector/knn/mlp -- all fit as sklearn
    Pipelines by nutpred.predict) is explained via the model-agnostic
    shap.Explainer over model.predict / predict_proba, since a Pipeline
    isn't a tree shap can introspect directly.
    """
    import shap

    if model_type in ("xgboost", "random_forest", "hist_gradient_boosting"):
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_target)
        if isinstance(sv, list):  # older shap API: per-class list for classifiers
            sv = sv[1] if len(sv) > 1 else sv[0]
        return np.asarray(sv)

    predict_fn = (lambda x: _predict_proba_positive(model, x)) if stage == "classifier" else model.predict
    rng = np.random.RandomState(0)
    n_bg = min(50, len(X_background))
    background = (
        X_background[rng.choice(len(X_background), n_bg, replace=False)]
        if len(X_background) > n_bg
        else X_background
    )
    explainer = shap.Explainer(predict_fn, background)
    sv = explainer(X_target)
    return np.asarray(sv.values if hasattr(sv, "values") else sv)


def _save_shap_summary_and_bar(shap_values: np.ndarray, X: np.ndarray, feature_cols: list, shap_dir: str, tag: str) -> None:
    import shap

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X, feature_names=feature_cols, show=False)
    plt.title(f"SHAP Summary: {tag}")
    plt.tight_layout()
    plt.savefig(os.path.join(shap_dir, f"shap_summary_{tag}.png"), dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, feature_names=feature_cols, plot_type="bar", show=False)
    plt.title(f"SHAP Feature Importance: {tag}")
    plt.tight_layout()
    plt.savefig(os.path.join(shap_dir, f"shap_bar_{tag}.png"), dpi=300, bbox_inches="tight")
    plt.close()

    mean_abs = np.abs(shap_values).mean(axis=0)
    grouped = pd.Series(mean_abs, index=[_feature_group(c) for c in feature_cols]).groupby(level=0).sum()
    grouped = grouped.sort_values(ascending=False)
    plt.figure(figsize=(8, 5))
    grouped.plot(kind="bar")
    plt.title(f"Mean |SHAP| by feature group: {tag}")
    plt.ylabel("Summed mean |SHAP| within group")
    plt.tight_layout()
    plt.savefig(os.path.join(shap_dir, f"shap_group_bar_{tag}.png"), dpi=300, bbox_inches="tight")
    plt.close()


def create_shap_plots(
    food_df_by_group: dict,
    models_dicts: dict,
    feature_sets_by_group: dict,
    outdir: str,
    model_type: str = "xgboost",
):
    """SHAP analysis for the FullNut models trained by nutpred.predict.train_tree_models.

    food_df_by_group: {group_name: DataFrame} -- pass the per-group dataframes
        (e.g. run.py's group_dfs), not a single shared one: binary/score/umap/
        aux/cluster feature columns are computed onto each group's own
        DataFrame, not the pre-group-split filtered_df.
    models_dicts: {group_name: {feature_set_name: {nutrient: {"regressor": model,
        "classifier": model_or_absent}}}} -- the actual shape train_tree_models
        returns (previously this function assumed a different, stale shape).
    feature_sets_by_group: {group_name: {feature_set_name: [col, ...]}} -- the
        feature-set column lists, per group (feature sets can differ slightly
        per group, e.g. via --aux-feature/--extra-feature-groups).
    """
    logger.info("Creating SHAP plots (model_type=%s)", model_type)

    shap_dir = os.path.join(outdir, "shap")
    os.makedirs(shap_dir, exist_ok=True)

    for group_name, models_dict in models_dicts.items():
        food_df = food_df_by_group.get(group_name)
        if food_df is None:
            logger.warning("No dataframe found for group %s, skipping SHAP", group_name)
            continue
        test_data = food_df[food_df["SampleType"] == "test"]
        train_data = food_df[food_df["SampleType"] == "train"]
        if len(test_data) == 0:
            logger.warning("No test data available for SHAP analysis in group %s", group_name)
            continue
        feature_sets = feature_sets_by_group.get(group_name, {})
        for feature_set_name, per_nutrient in models_dict.items():
            feature_cols = feature_sets.get(feature_set_name)
            if not feature_cols:
                logger.warning("No feature columns found for %s/%s, skipping SHAP", group_name, feature_set_name)
                continue

            # Explain every test row, with missing values filled by the TRAIN
            # median exactly as the models saw them in training.
            from sklearn.impute import SimpleImputer
            X_train_raw = train_data[feature_cols].to_numpy(float)
            imputer = SimpleImputer(strategy="median", keep_empty_features=True).fit(X_train_raw)
            X_train_full = imputer.transform(X_train_raw)
            X_test_clean = imputer.transform(test_data[feature_cols].to_numpy(float))

            for nutrient, stage_models in per_nutrient.items():
                for stage, model in (("regressor", stage_models.get("regressor")), ("classifier", stage_models.get("classifier"))):
                    if model is None:
                        continue
                    tag = f"{nutrient}_{feature_set_name}_{group_name}_{stage}"
                    if isinstance(model, DummyClassifier):
                        logger.info("Skipping SHAP for %s: DummyClassifier fallback has no feature attributions", tag)
                        continue
                    try:
                        sv = _shap_values_for(model_type, model, stage, X_train_full, X_test_clean)
                        _save_shap_summary_and_bar(sv, X_test_clean, feature_cols, shap_dir, tag)
                        logger.info("SHAP plots saved for %s", tag)
                    except Exception as e:
                        logger.warning("SHAP failed for %s: %s", tag, e)

    logger.info("All SHAP plots saved to %s", shap_dir)

__all__ = ["compare_feature_sets", "create_scatterplots", "create_shap_plots"]