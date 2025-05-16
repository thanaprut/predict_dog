def predict_breed(CLIENT, num_run=5, file_obj=None):
    import concurrent.futures

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
