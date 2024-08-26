import os
import json
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.multiprocessing import Process, set_start_method
import torch.distributed as dist
import openai
import asyncio
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
from tqdm import tqdm
from PIL import Image  # Add this import


# Set your OpenAI API key from environment variable
openai.api_key = ""

class ChatGPTDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_item = self.data[idx]
        filenames = data_item['image']
        original_question = data_item['question']
        gpt_value = data_item['answer']
        image_id = data_item['id']

        return filenames, original_question, gpt_value, image_id

def openai_chat_completion(prompt):
    return openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150,
        temperature=0.2,
        top_p=0.1
    )

async def fetch_openai_response(prompt):
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(None, openai_chat_completion, prompt)
    return response.choices[0].message['content'].strip()

def worker(rank, world_size, args, data_dict):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = str(rank)

    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')

    with open(args.data, 'r') as f:
        data_all = json.load(f)

    dataset = ChatGPTDataset(data_all)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)  # Set num_workers to 0

    # Load image captioning model and processor
    caption_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning").to(device)
    feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    caption_tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    model = DDP(caption_model, device_ids=[rank])
    async def process_batch(batch, pbar):
        for filenames, original_question, gpt_value, image_id in zip(*batch):
            images = []
            update_id = image_id.item()
            for filename in filenames:
                # Load the image from file
                image = Image.open(filename).convert("RGB")
                images.append(image)

            # Preprocess for caption generation
            inputs = feature_extractor(images=images, return_tensors="pt").to(device)

            # Generate a caption
            caption_ids = caption_model.generate(**inputs)
            caption = caption_tokenizer.decode(caption_ids[0], skip_special_tokens=True)

            # Define a question about the image
            prompt = "What are the important objects in the current scene? Those objects will be considered for the future reasoning and driving decision."
            additional_prompt = "Please analyze the risk of the referred object step by step"
            final_question = prompt + additional_prompt
            visual_llama_prompt = f"""
            You are Visual LLaMA, a model trained to understand and analyze images and text together. Given the following image caption and question, provide a detailed answer.
            Image Caption: {caption}
            Question: {original_question}
            Step-by-Step Analysis: {final_question}
            """
            result = await fetch_openai_response(visual_llama_prompt)
            data_dict.append({
            'id': update_id,
            'question': original_question,
            'gt_answer': gpt_value,
            'answer': result,
            })
            pbar.update(1)
            print(f"GPU {rank}: Result - {result}")

    async def main_processing():
        tasks = []
        with tqdm(total=len(dataloader), desc=f"GPU {rank}") as pbar:
            for batch in dataloader:
                tasks.append(process_batch(batch, pbar))
            await asyncio.gather(*tasks)

    asyncio.run(main_processing())

    print(f"GPU {rank} finished")
    dist.destroy_process_group()



def main():
    parser = argparse.ArgumentParser(description='ChatGPT Adapter')
    parser.add_argument('--data', type=str, default="test_llama.json", help='path to test data')
    parser.add_argument('--output', type=str, default="output.json", help='path to output file')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size for parallel processing')
    parser.add_argument('--num_gpus', type=int, default=8, help='number of gpus to use')
    args = parser.parse_args()

    num_gpus = args.num_gpus
    print(f"Using {num_gpus} GPUs")

    manager = torch.multiprocessing.Manager()
    data_dict = manager.list()
    processes = []

    for rank in range(num_gpus):
        p = Process(target=worker, args=(rank, num_gpus, args, data_dict))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    with open(args.output, "w") as f:
        json.dump(list(data_dict), f, indent=4)

if __name__ == '__main__':
    set_start_method('spawn')
    main()
