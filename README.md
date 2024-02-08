# Alarm Flood Classification

<link to thesis>

Implements papaers from alarm flood classification and clustering research. Clustering methods are modified to do classification, descriptions of the modifications in the thesis. References to the corresponding papers are given at the beginning of .py files in the `code/methods` folder

This project is tested with Python 3.7. The notebook `test_methods.ipynb` provides examples of how to run the methods implemented in this project.

## Data Requirements

Input data should be in CSV format which contains a table with the following columns:

- `flood_id`: The identifier for the alarm flood.
- `alarmNumber`: The number of the alarm.
- `startTimestamp`: The start timestamp of the alarm, given in milliseconds from the start of the first alarm.
- `endTimestamp`: The end timestamp of the alarm, given in milliseconds from the start of the first alarm.

Corresponding labels should be in a CSV file named `{system}.csv` located in the `data/ground_truth/` directory.
The file contains one column where each row has a int label for the corresponding alarm flood

## Running the Code

To run the code, open the `test_methods.ipynb` notebook and follow the instructions provided in the notebook.
`notebooks/` folder also contains addional notebooks for visaulising and analysing results
Note that some of the notebooks which process or visualize the raw alarm data can be quite specific and should only be used as reference.

Running CASTLE requires cloning repo from https://github.com/ChangWeiTan/MultiRocket to adjacent folder and either installing required libraries or (adjusting files so they are used as they are not required for using the importet functions)

![Depiction of online alarm flood classification in real world](images/DALL·E_alarm_flood_scene.png)
