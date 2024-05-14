import pickle
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class Dataset:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series

    def get_sample(
        self, sample_size_train: int = 10000, sample_size_test: int = 2000
    ):
        idxs_train = random.sample(
            self.X_train.index.tolist(), sample_size_train
        )
        idxs_test = random.sample(self.X_test.index.tolist(), sample_size_test)
        return Dataset(
            X_train=self.X_train.loc[idxs_train, :],
            X_test=self.X_test.loc[idxs_test, :],
            y_train=self.y_train.loc[idxs_train],
            y_test=self.y_test.loc[idxs_test],
        )

    def get_shapes(self):
        return {
            "X_train": self.X_train.shape,
            "X_test": self.X_test.shape,
            "y_train": self.y_train.shape,
            "y_test": self.y_test.shape,
        }


class Pipeline:
    def __init__(
        self,
        data_path: str = Path(__file__).parents[1] / "WEC2022_Data",
        model_type: str = "predict_upgrade",
    ):
        self.target = (
            "UPGRADE_SALES_DATE"
            if model_type == "predict_when_upgrade"
            else "UPGRADED_FLAG"
        )
        self.data_path = data_path
        self.model_type = model_type
        self.scaler = StandardScaler()
        self.df = self.merge_files()

    def read_csv_file(self, file_prefix) -> pd.DataFrame:
        return pd.read_csv(
            self.data_path / f"{file_prefix}_train.csv", sep=";"
        )

    def read_all_files(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        bkg = self.read_csv_file("BKG").drop(columns=self.get_bkg_drop_cols())
        logger.info("loaded booking data")
        tkt = self.read_csv_file("TKT")
        logger.info("loaded ticket data")
        fcp = self.read_csv_file("FCP")
        logger.info("loaded flight coupon data")
        return (
            bkg,
            tkt,
            fcp[fcp.UPGRADED_FLAG == "Y"]
            if self.model_type == "predict_when_upgrade"
            else fcp,
        )

    @staticmethod
    def get_bkg_drop_cols() -> list:
        return ["UPGRADED_FLAG", "UPGRADE_TYPE", "UPGRADE_SALES_DATE"]

    def merge_files(self) -> pd.DataFrame:
        bkg, tkt, fcp = self.read_all_files()
        logger.info(f"{datetime.now()} merging datasets")
        return fcp.merge(tkt, on="TICKET_NUMBER", how="left").merge(
            bkg, on="BOOKING_ID", how="left"
        )

    @staticmethod
    def discard_nonupgraded_rows(fcp: pd.DataFrame) -> pd.DataFrame:
        return fcp[fcp["UPGRADED_FLAG"] == "Y"]

    def get_columns_to_drop(self) -> list:
        cols_to_drop = [
            "TICKET_NUMBER",
            "ORIGIN_AIRPORT_CODE",
            "DESTINATION_AIRPORT_CODE",
            "SALES_DATE",
            "FLIGHT_DATE_LOCAL",
            "MARKETING_CARRIER",
            "OPERATIONAL_CARRIER",
            "BOOKED_CLASS",
            "AIRCRAFT_TYPE",
            "FARE_BASIS",
            "BOOKING_ID",
            "ORIGINAL_TICKET_NUMBER",
            "SEGMENTS",
            "FLIGHT_COUPONS",
            "FORM_OF_PAYMENT",
            "CURRENCY",
            "TOTAL_PRICE",
            "LOYAL_CUSTOMER_ID",
            "LOYAL_CUSTOMER_DATE_OF_BIRTH",
            "LOYAL_CUSTOMER_REGISTERED_DATE",
            "SALES_DATE",
            "SALES_MARKET",
            "SEGMENTS",
            "INTINERARY",
            "BOOKING_ORIGIN_AIRPORT",
            "BOOKING_ORIGIN_COUNTRY_CODE",
            "BOOKING_DEPARTURE_TIME_UTC",
            "BOOKING_DESTINATION_AIRPORT",
            "BOOKING_DESTINATION_COUNTRY_CODE",
            "BOOKING_ARRIVAL_TIME_UTC",
            "UPGRADE_TYPE",
        ]
        cols_to_drop.append(
            "UPGRADE_SALES_DATE"
        ) if self.model_type == "predict_upgrade" else None
        return cols_to_drop

    def prepare_new_cols_for_predict_when_model(self) -> None:
        self.df["purchase_time_diff"] = (
            self.df["UPGRADE_SALES_DATE"]
            - self.df["BOOKING_DEPARTURE_TIME_UTC"]
        )
        self.df["purchase_time_diff"] = self.df["purchase_time_diff"].apply(
            lambda x: x.days
        )
        logger.info("prepared new cols for predict when model")

    @staticmethod
    def get_sus_air_type() -> list:
        return ["763", "788", "789", "332", "787"]

    @staticmethod
    def get_sus_currency() -> list:
        return ["JPY", "USD", "CAD", "SGD", "VND", "AED"]

    @staticmethod
    def get_sus_payment() -> list:
        return [
            "UNION",
            "CCDS6",
            "CCSW9",
            "NET R",
            "CCJC3",
            "CCAX3",
            "BARTE",
            "CCVI4",
            "PAY24",
        ]

    def filter_wrong_ticket_prices(self) -> None:
        self.df = self.df[self.df["TOTAL_PRICE_PLN"] > 0]
        logger.info("filtered wrong ticket prices")

    @staticmethod
    def get_datetime_columns() -> list:
        return [
            "UPGRADE_SALES_DATE",
            "BOOKING_DEPARTURE_TIME_UTC",
            "FLIGHT_DATE_LOCAL",
            "SALES_DATE",
            "BOOKING_ARRIVAL_TIME_UTC",
            "TIME_DEPARTURE_LOCAL_TIME",
        ]

    def convert_datetime_columns_to_pandas_format(self) -> None:
        self.df[self.get_datetime_columns()] = self.df[
            self.get_datetime_columns()
        ].apply(pd.to_datetime)
        logger.info("converted datetime columns to pandas format")

    def filter_wrong_booking_window(self) -> None:
        self.df = self.df[self.df["BOOKING_WINDOW_D"] != -1]
        logger.info("filtered wrong booking windows")

    def filter_nonpositive_flight_distance(self) -> None:
        self.df = self.df[self.df["FLIGHT_DISTANCE"] >= 0]
        logger.info("filtered nonpositive flight distance")

    def get_sale_to_flight_time(self) -> None:
        self.df["sale_to_flight_time"] = (
            self.df["FLIGHT_DATE_LOCAL"] - self.df["SALES_DATE"]
        )
        self.df["sale_to_flight_time"] = self.df["sale_to_flight_time"].apply(
            lambda x: x.days
        )
        logger.info("got sale to flight time col")

    @staticmethod
    def get_stay_length_map() -> dict:
        return {-9999: np.nan}

    def map_stay_length(self) -> None:
        self.df["STAY_LENGTH_D"] = self.df["STAY_LENGTH_D"].replace(
            self.get_stay_length_map()
        )

    def get_flight_len(self) -> None:
        self.df["flight_len"] = (
            self.df["BOOKING_ARRIVAL_TIME_UTC"]
            - self.df["BOOKING_DEPARTURE_TIME_UTC"]
        )
        self.df["flight_len"] = self.df["flight_len"].apply(
            lambda x: x.seconds / 3600
        )
        logger.info("got flight len col")

    def map_departure_time_to_hours(self) -> None:
        self.df["TIME_DEPARTURE_LOCAL_TIME"] = self.df[
            "TIME_DEPARTURE_LOCAL_TIME"
        ].apply(lambda x: x.hour)
        logger.info("mapped departure time to hours")

    @staticmethod
    def get_yes_no_binary_map() -> dict:
        return {"Y": 1, "N": 0}

    def check_for_add_upgrade(self) -> None:
        emd = self.read_csv_file("EMD")
        self.df["if_additional_upgrade"] = np.where(
            np.isin(
                self.df["TICKET_NUMBER"],
                emd["REFERENCE_TICKET_NUMBER"].unique(),
            ),
            1,
            0,
        )
        logger.info("checked for add upgrade")

    def check_for_same_carrier(self) -> None:
        self.df["same_carrier"] = np.where(
            self.df["MARKETING_CARRIER"] == self.df["OPERATIONAL_CARRIER"],
            1,
            0,
        )
        logger.info("checked for same carrier")

    def check_for_sus_aircraft(self) -> None:
        self.df["is_sus_aircraft"] = np.where(
            np.isin(self.df["AIRCRAFT_TYPE"], self.get_sus_air_type()), 1, 0
        )
        logger.info("checked for sus aircraft")

    def map_target_variable_for_training(self) -> None:
        self.df["UPGRADED_FLAG"] = self.df["UPGRADED_FLAG"].map(
            self.get_yes_no_binary_map()
        )

    def check_for_sus_payment(self) -> None:
        self.df["is_sus_payment"] = np.where(
            np.isin(self.df["FORM_OF_PAYMENT"], self.get_sus_payment()), 1, 0
        )
        logger.info("checked for sus payment")

    def get_intinerary_len(self) -> None:
        self.df["intinerary_len"] = self.df["INTINERARY"].apply(
            lambda x: len(x.split("-"))
        )
        logger.info("got itinerary len col")

    def check_for_sus_currency(self) -> None:
        self.df["is_sus_currency"] = np.where(
            np.isin(self.df["FORM_OF_PAYMENT"], self.get_sus_currency()), 1, 0
        )
        logger.info("checked for sus currency")

    @staticmethod
    def get_map_gender() -> dict:
        return {"M": 1, "F": 0}

    def map_genders(self) -> None:
        self.df["PAX_GENDER"] = self.df["PAX_GENDER"].map(
            self.get_map_gender()
        )
        logger.info("mapped genders")

    def map_corporate_contract_flg(self) -> None:
        self.df["CORPORATE_CONTRACT_FLG"] = self.df[
            "CORPORATE_CONTRACT_FLG"
        ].map(self.get_yes_no_binary_map())
        logger.info("mapped corporate contract flag")

    def map_loyal_customer(self) -> None:
        self.df["LOYAL_CUSTOMER"] = self.df["LOYAL_CUSTOMER"].map(
            self.get_yes_no_binary_map()
        )
        logger.info("mapped loyal customers")

    def map_booking_long_houl_flag(self) -> None:
        self.df["BOOKING_LONG_HOUL_FLAG"] = self.df[
            "BOOKING_LONG_HOUL_FLAG"
        ].map(self.get_yes_no_binary_map())
        logger.info("mapped booking long houl flag")

    def map_booking_domestic_flag(self) -> None:
        self.df["BOOKING_DOMESTIC_FLAG"] = self.df[
            "BOOKING_DOMESTIC_FLAG"
        ].map(self.get_yes_no_binary_map())
        logger.info("mapped booking domestic flag")

    def clean_df(self):
        logger.info(f"{datetime.now()} cleaning df ...")
        self.convert_datetime_columns_to_pandas_format()
        self.prepare_new_cols_for_predict_when_model() if self.model_type == "predict_when_upgrade" else None
        self.filter_wrong_ticket_prices()
        self.filter_wrong_booking_window()
        self.filter_nonpositive_flight_distance()
        self.get_sale_to_flight_time()
        self.get_flight_len()
        self.map_departure_time_to_hours()
        self.check_for_add_upgrade()
        self.check_for_same_carrier()
        self.check_for_sus_aircraft()
        self.map_target_variable_for_training()
        self.check_for_sus_payment()
        self.get_intinerary_len()
        self.check_for_sus_currency()
        self.map_genders()
        self.map_corporate_contract_flg()
        self.map_loyal_customer()
        self.map_booking_long_houl_flag()
        self.map_booking_domestic_flag()
        self.df.drop(columns=self.get_columns_to_drop(), inplace=True)
        self.df.dropna(inplace=True)
        logger.info(f"dropped unnecessary columns. finished cleaning df")

    @staticmethod
    def get_oh_cols() -> list:
        return [
            "FLIGHT_RANGE",
            "BOOKED_CABIN",
            "VAB",
            "PAX_TYPE",
            "SALES_CHANNEL",
            "TRIP_TYPE",
        ]

    def get_oh_encoding(self) -> pd.DataFrame:
        logger.info(f"{datetime.now()} getting OH encoding")
        return pd.get_dummies(
            self.df[self.get_oh_cols()],
            prefix=self.get_oh_cols(),
            drop_first=True,
        )

    def concat_df_with_oh_encoding(self) -> None:
        self.clean_df()
        self.df = pd.concat([self.df, self.get_oh_encoding()], axis=1).drop(
            columns=self.get_oh_cols()
        )

    def scale_final_dataset(self, save_path: Path = None) -> Dataset:
        self.concat_df_with_oh_encoding()
        X_train, X_test, y_train, y_test = train_test_split(
            self.df.drop(columns=self.target),
            self.df[self.target],
            test_size=0.2,
            random_state=69420,
            stratify=self.df[self.target]
            if self.model_type == "predict_upgrade"
            else None,
        )
        X_train[self.get_cols_to_scale()] = self.scaler.fit_transform(
            X_train[self.get_cols_to_scale()]
        )
        X_test[self.get_cols_to_scale()] = self.scaler.transform(
            X_test[self.get_cols_to_scale()]
        )
        logger.info("scaled features")

        dataset = Dataset(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
        )

        if save_path:
            with open(save_path, "wb") as file:
                pickle.dump(dataset, file, pickle.HIGHEST_PROTOCOL)

        return dataset

    def get_cols_to_scale(self) -> list:
        cols_to_scale = [
            "COUPON_NUMBER",
            "FLIGHT_DISTANCE",
            "TOTAL_PRICE_PLN",
            "BOOKING_WINDOW_D",
            "STAY_LENGTH_D",
            "PAX_N",
            "sale_to_flight_time",
            "flight_len",
            "intinerary_len",
        ]
        cols_to_scale.append(
            "purchase_time_diff"
        ) if self.model_type == "predict_when_upgrade" else None
        return cols_to_scale


if __name__ == "__main__":
    data_path = Path(__file__).parents[1] / "data" / "raw_data"
    d = Pipeline(model_type="predict_upgrade", data_path=data_path)
    # d.concat_df_with_oh_encoding()
    # d.df = pd.read_csv(data_path).drop(columns="Unnamed: 0")
    # save_path_csv = data_path.parents[0] / "dataset.csv"
    # d.df.to_csv(save_path_csv, index=False)
    # this is how you access df if you need it for k-fold, plain df without splitting
    save_path = data_path.parents[0] / "dataset.pickle"
    dataset_split_sets = (
        d.scale_final_dataset(save_path=save_path),
    )  # this is how you access df if you don't do kfold cross-validation
