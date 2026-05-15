import pandas as pd

IN_CSV  = "output_image_1/validated_valve_pipe_links_image_1_New.csv"
OUT_CSV = "final_connections_image_1_New.csv"

df = pd.read_csv(IN_CSV)

final = df[df.status == "ACCEPT"].copy()

final.to_csv(OUT_CSV, index=False)

print(f"✅ Final connections saved")
print(f"→ {OUT_CSV}")
print(f"Rows: {len(final)}")


# import pandas as pd

# df = pd.read_csv("output_image_1/validated_valve_pipe_links_image_1.csv")

# print(df["status"].value_counts())
# print(df["valve_type"].value_counts())




# import pandas as pd

# df = pd.read_csv("output_image_1/validated_valve_pipe_links_image_1.csv")

# final = df[df.status == "ACCEPT"].copy()

# final.to_csv("final_connections_image_1.csv", index=False)

# print("FINAL ROWS:", len(final))
