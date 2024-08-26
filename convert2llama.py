import numpy as np
import json
import os


# Base directory
base_dir = "llama_adapter_v2_multimodal7b/data/nuscenes"

def convert2llama(root, dst):
    with open(root, 'r') as f:
        test_file = json.load(f)

    output = []
    images_exist = []
    images_notexist = []
    id_counter = 0
    
    for scene in test_file:
        for frame in scene:
            update_image = []
            image_paths = scene['images']
            # Change the path of the images
            for image in image_paths:
                image_path = os.path.join(base_dir, image)
                if os.path.exists(image_path):
                    update_image.append("data/nuscenes/" + image)
                    images_exist.append(image)
                else:
                    images_notexist.append(image)
            
            frame_data_question = scene['text_in']
            frame_data_answer = scene['text_out']
            
            if len(update_image) > 3 and len(output) < 1000:
                output.append(
                    {
                        "image": update_image,
                        "question": frame_data_question,
                        "answer": frame_data_answer,
                        "id": id_counter
                    }
                )
                id_counter += 1
            
    print(len(images_exist))
    print(len(images_notexist))
    print(len(output))
    
    with open(dst, 'w') as f:
        json.dump(output, f, indent=4)

if __name__ == '__main__':
    root = "reason2drive.json"
    dst = "test_gpt.json"
    convert2llama(root, dst)