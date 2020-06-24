import csv
import pandas
import logging


class Output:
    temp_dir = "model/output_temp/"

    temp_names = []

    def save_as_csv(self, name="", solution=[]):
        """Save solution as a csv file"""
        keys = list(solution[0].keys())
        values = []
        for line in solution:
            values.append(list(line.values()))

        self.temp_names.append(name)

        path = self.temp_dir + name + ".csv"

        with open(path, 'w+', newline='', encoding='utf-16') as file:
            writer = csv.writer(file)
            writer.writerow(keys)
            for line in values:
                writer.writerow(line)

        return 0

    def store_output(self, filename="output.xlsx"):

        # open temp output and load all files to results_dict
        results_dict = {}
        for name in self.temp_names:
            solution = []
            with open('model/output_temp/' + name + '.csv', newline='', encoding='utf-16') as file:
                reader = csv.DictReader(file)
                for line in reader:
                    solution.append(line)
            results_dict[name] = solution

        # write final output
        writer = pandas.ExcelWriter(filename)
        if len(results_dict) == 0:
            logging.warning("NO SOLUTION FOUND, NOTHING TO BE SAVED")
            return

        for sheet_name in results_dict.keys():
            results = results_dict[sheet_name]
            if results is None or len(results) == 0:
                break
            df = pandas.DataFrame(results)
            df.to_excel(writer, sheet_name=sheet_name, columns=[*results[0].keys()])
        writer.save()


if __name__ == "__main__":
    pass
