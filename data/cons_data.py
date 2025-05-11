import json
import wrds
import os


class get_cons:
    def __init__(self, etf):
        self.etf = etf
        self.cons = None
        self.out_file = "cons_by_date.json"

    def query_db(self):
        db = wrds.Connection()
        self.cons = db.raw_sql(
            f"""
            SELECT 
                as_of_date, 
                composite_ticker, 
                constituent_ticker, 
                weight
            FROM 
                etfg_constituents.constituents
            WHERE 
                composite_ticker = '{self.etf}'
                AND (
                    -- Match quarter-end dates
                    (EXTRACT(MONTH FROM as_of_date) = 3 AND EXTRACT(DAY FROM as_of_date) = 31) OR  -- March 31
                    (EXTRACT(MONTH FROM as_of_date) = 6 AND EXTRACT(DAY FROM as_of_date) = 30) OR  -- June 30
                    (EXTRACT(MONTH FROM as_of_date) = 9 AND EXTRACT(DAY FROM as_of_date) = 30) OR  -- September 30
                    (EXTRACT(MONTH FROM as_of_date) = 12 AND EXTRACT(DAY FROM as_of_date) = 31)     -- December 31
                )
            ORDER BY 
                as_of_date;

            """,
            date_cols=["as_of_date"],
        )

        self.cons = self.cons.groupby(
            ["as_of_date", "constituent_ticker"], as_index=False
        ).last()

    def output(self):
        tickers_dictionary = (
            self.cons.groupby("as_of_date")["constituent_ticker"].apply(list).to_dict()
        )

        tickers_dictionary = {
            key.strftime("%Y-%m-%d"): value for key, value in tickers_dictionary.items()
        }

        with open(self.out_file, "w") as json_file:
            json.dump(tickers_dictionary, json_file, default=str)

    def run(self):
        self.query_db()
        self.output()

    def read(
        self,
    ):
        if os.path.exists(f"./{self.out_file}"):
            with open(self.out_file, "r") as json_file:
                data = json.load(json_file)
                return data
        else:
            print("no file")
