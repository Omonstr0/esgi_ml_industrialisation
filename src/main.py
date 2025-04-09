import pandas as pd
from sklearn.linear_model import Ridge

def make_predictions(config):
    model = config["model"]
    if model == "PrevMonthSale":
        return PrevMonthSale(config)
    if model == "SameMonthLastYearSales":
        return SameMonthLastYearSales(config)
    if model == "Ridge":
        return AutoRegressiveModel(config)

def PrevMonthSale(config):
    df_sales = pd.read_csv(config["data"]["sales"])

    df_sales["prediction"] = df_sales.groupby("item_id")["sales"].shift(1)

    df_sales = df_sales[df_sales["dates"] >= config["start_test"]].reset_index(drop=True)

    return df_sales[["dates", "item_id", "prediction"]]

def SameMonthLastYearSales(config):
    df_sales = pd.read_csv(config["data"]["sales"])

    df_sales["prediction"] = df_sales.groupby("item_id")["sales"].shift(12)

    df_sales = df_sales[df_sales["dates"] >= config["start_test"]].reset_index(drop=True)

    return df_sales[["dates", "item_id", "prediction"]]

def build_features(config):
    df = pd.read_csv(config["data"]["sales"])

    # Convertir les dates en datetime pour l'ordre
    df["dates"] = pd.to_datetime(df["dates"])

    # Cas spécial : uniquement past_sales
    if config["features"] == ["past_sales"]:
        df = df.sort_values(["item_id", "dates"])
        df["past_sales"] = df.groupby("item_id")["sales"].shift(1)
    else:
        # Créer tous les lags nécessaires
        df = df.sort_values(["item_id", "dates"])
        for lag in range(1, 16):
            df[f"sales_t_{lag}"] = df.groupby("item_id")["sales"].shift(lag)

        df["same_month_last_year"] = df["sales_t_12"]
        df["avg_last_year"] = df[[f"sales_t_{i}" for i in range(1, 13)]].mean(axis=1)
        df["growth_q"] = df[[f"sales_t_{i}" for i in range(1, 4)]].mean(axis=1) / df[[f"sales_t_{i}" for i in range(13, 16)]].mean(axis=1)
        df["seasonality"] = df["sales_t_12"] * df["growth_q"]

        if "past_sales" in config["features"]:
            df["past_sales"] = df["sales_t_1"]

    # Merge des autres fichiers (marketing, etc.)
    for feature in config["features"]:
        if feature not in df.columns and feature in config["data"]:
            df_extra = pd.read_csv(config["data"][feature])
            df_extra["dates"] = pd.to_datetime(df_extra["dates"])
            df = df.merge(df_extra, on=["dates", "item_id"], how="left")

    df = df.dropna()

    return df

def AutoRegressiveModel(config):
    df = build_features(config)

    df_train = df[df["dates"] < config["start_test"]]
    df_test = df[df["dates"] >= config["start_test"]]

    features = config["features"]

    model = Ridge()
    model.fit(df_train[features], df_train["sales"])

    df_test = df_test.copy()
    df_test["prediction"] = model.predict(df_test[features])

    return df_test[["dates", "item_id", "prediction"]]