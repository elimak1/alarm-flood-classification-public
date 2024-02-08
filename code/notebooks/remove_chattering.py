import pandas as pd


def merge_chattering_alarm(df, timewindow):

    prev = None

    processed_df = pd.DataFrame(columns=df.columns)
    for i, row in df.iterrows():
        if prev is None:
            prev = row
            continue

        if (row["startTimestamp"] - prev["endTimestamp"]).total_seconds() < timewindow*60:
            prev["endTimestamp"] = row["endTimestamp"]
        else:
            processed_df = processed_df.append(prev)
            prev = row
    
    processed_df = processed_df.append(prev)
    return processed_df

def remove_nuisance_alarms(alarms, timewindow, levels_to_filter = [], codes_to_filter = [], messages_to_keep = []):
    df = alarms.copy()
    df = df[(~df["level"].isin(levels_to_filter)) | (df["alarmNumber"].isin(messages_to_keep))]
    df = df[~df["alarmNumber"].isin(codes_to_filter)]
    df["alarmId"] = df["deviceId"] + "_" + df["alarmNumber"]

    return df.groupby("alarmId").apply(lambda x: merge_chattering_alarm(x, timewindow))