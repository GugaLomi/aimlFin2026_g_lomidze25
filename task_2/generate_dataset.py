import pandas as pd
import random

events = [
    "LOGIN",
    "LOGOUT",
    "FILE_ACCESS",
    "PRIV_ESC",
    "DATA_EXFIL",
    "FAILED_LOGIN",
    "PASSWORD_RESET",
    "NONE"
]

rows = []

for _ in range(600):

    attack_type = random.choices(
        population=[0,1,2,3],
        weights=[60,15,15,10]
    )[0]

    if attack_type == 0:
        sequence = random.choices(
            ["LOGIN","FILE_ACCESS","LOGOUT","NONE"],
            k=6
        )

    elif attack_type == 1:
        sequence = ["LOGIN","PRIV_ESC","FILE_ACCESS","LOGOUT"]
        sequence += random.choices(events, k=2)

    elif attack_type == 2:
        sequence = ["LOGIN","PRIV_ESC","DATA_EXFIL"]
        sequence += random.choices(events, k=3)

    elif attack_type == 3:
        sequence = ["FAILED_LOGIN","FAILED_LOGIN","FAILED_LOGIN"]
        sequence += random.choices(events, k=3)

    sequence = sequence[:6]
    rows.append(sequence + [attack_type])

df = pd.DataFrame(rows)
df.to_csv("cyber_multiclass_dataset.csv", index=False, header=False)

print("Dataset generated successfully!")
