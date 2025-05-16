
# import os
# image_file = os.path.join("predicts", "10.jpg")
# image = cv2.imread(image_file)
import tempfile

def predict_breed(CLIENT, num_run=5, file_obj=None):
    import concurrent.futures
  
    # temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    # temp_file.write(file_obj.read())
    # temp_file.close()
    def infer_once():
        return CLIENT.infer(file_obj, model_id="stanford-dogs-0pff9/3")

    all_predictions = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(infer_once) for _ in range(num_run)]
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                all_predictions.extend(result.get("predictions", []))
            except Exception as e:
                print(f"Error in inference: {e}")
    print(all_predictions)
    breed_groups = {}
    for pred in all_predictions:
        breed = pred["class"]
        if breed not in breed_groups:
            breed_groups[breed] = []
        breed_groups[breed].append(pred)
    best_predictions = [
        max(group, key=lambda x: x["confidence"])
        for group in breed_groups.values()
    ]
    print("📦 Total predictions:", len(all_predictions))
    return best_predictions


# if result:
#     labels = [item["class"] for item in result["predictions"]]
#     detections = sv.Detections.from_inference(result)
#     bounding_box_annotator = sv.BoxAnnotator()
#     label_annotator = sv.LabelAnnotator(text_position=sv.Position.TOP_LEFT, text_thickness = 3, text_color=sv.Color.BLACK, color=sv.ColorPalette.ROBOFLOW, text_scale=5.0, text_padding=10)
#     annotated_image = bounding_box_annotator.annotate(scene=image, detections=detections)
#     annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels = labels)

#     def classify_owner(row):
#         if row["Energy Level"] >= 4 and row["Exercise Needs"] >= 4:
#             return "B - Active Lifestyle"
#         elif row["Tendency To Bark Or Howl"] <= 2 and row["Dog Size"] == "Small":
#             return "C - Family with Kids"
#         elif row["Energy Level"] <= 2:
#             return "A - Stay-at-Home"
#         else:
#             return "D - General Owner"
        
#     name_breed = result["predictions"][0]["class"].strip()

        
#     df = pd.read_csv("data/dogs_cleaned.csv")
#     df["Suitable_For"] = df.apply(classify_owner, axis=1)

#     df_get = df.loc[df["Breed Name"] == name_breed]
#     print(df_get)
#     mask = df["Breed Name"].str.lower().eq(name_breed.lower())

#     if mask.any():                 
#         df_breed = df.loc[mask].iloc[0].copy()
#         print(df_breed)
#     else:
#         print(f"ไม่พบพันธุ์สุนัขชื่อ '{name_breed}' ในชุดข้อมูล")

#     resized_image = cv2.resize(image, (614, 614))
#     cv2.imshow("Resized", resized_image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
