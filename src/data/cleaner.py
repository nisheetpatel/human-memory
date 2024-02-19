import os
from dataclasses import dataclass

import dateutil.parser


@dataclass
class DataCleaner:
    """Clean the raw data folder to get rid of useless files."""

    path: str
    ref_date: str = "2000-01-01"
    min_size_bytes: int = 10_000

    def __post_init__(self):
        self.csv_files = [
            file for file in os.listdir(self.path) if file.endswith(".csv")
        ]

    def get_date(self, file: str) -> str:
        try:
            return file.split("_")[3]
        except dateutil.parser.ParserError:
            print(f"Warning: could not parse date from {file}; removed.")
            return self.ref_date

    @staticmethod
    def check_date_less_than(date: str, ref_date: str) -> list:
        date = dateutil.parser.parse(date)
        ref_date = dateutil.parser.parse(ref_date)
        return date < ref_date

    def is_not_csv(self, file: str) -> bool:
        return file not in self.csv_files

    def is_old(self, file: str) -> bool:
        return self.check_date_less_than(self.get_date(file), self.ref_date)

    def is_small(self, file: str) -> bool:
        return os.path.getsize(self.path + file) < self.min_size_bytes

    def clean(self) -> None:
        counter = 0
        for file in os.listdir(self.path):
            if self.is_not_csv(file) or self.is_old(file) or self.is_small(file):
                os.remove(self.path + file)
                print(f"Removed {file}.")
                counter += 1

        print(f"\nCleaned {self.path}:")
        print(f"Removed {counter} files. {len(os.listdir(self.path))} remain.\n")
