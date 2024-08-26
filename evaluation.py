#Import various libraries and tools needed for the evaluation and optimization tasks
#Including libraries for handing data, running machine learning models, and interacting
#with the GPT-3 model
import re
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import language_evaluation
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW
import sys
sys.path.append(".")
from gpt_eval import GPTEvaluation


#Added RewardModel class
#This part defines a simple reward model that uses a pre-trained GPT-2 model to generate hidden states
#and then passes these states through a linear layer to compute rewards.
class RewardModel(nn.Module):
    def __init__(self):
        super(RewardModel, self).__init__()
        #Loads a pre-trained GPT-2 model with hidden state outputs enabled.
        self.model = GPT2LMHeadModel.from_pretrained('gpt2', output_hidden_states=True)
        #Adds a linear layer to convert hidden states to a signle reward value
        self.linear = nn.Linear(self.model.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids, attention_mask=attention_mask) #Passes input data through GPT-2
        hidden_states = outputs.hidden_states[-1]  # Extracts the last layer hidden states
        rewards = self.linear(hidden_states[:, -1, :]) # Computes rewards using the linear layer
        return rewards


class evaluation_suit():
    #Initialize the evaluation suit, which includes different evaluators and the reward model its optimizer
    def __init__(self):
        # Sets up a language evaluator with BLEU, ROUGE_L, and CIDEr metrics.
        self.language_eval = language_evaluation.CocoEvaluator(coco_types=["BLEU", "ROUGE_L", "CIDEr"])
        # Initializes the custom ChatGPT evaluator.
        self.chatgpt_eval = GPTEvaluation()
        # Initializes data structures to store evaluation data.
        self.GPT = []
        self.accuracy = {"answer": [], "GT": []}
        self.language = {"answer": [], "GT": []}
        self.match = {"match": {"answer": [], "GT": []}, "GPT": []}

        # Initialize reward model and optimizer
        self.reward_model = RewardModel()
        self.optimizer = AdamW(self.reward_model.parameters(), lr=1e-5)  # Sets up an optimizer for the reward model
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')  # Initializes the tokenizer for GPT-2

    # Defines the method to evaluate accuracy
    def eval_acc(self):
        scores = []
        for i in range(len(self.accuracy["answer"])):
            answer = self.accuracy["answer"][i] #Retrieves the predicted answer
            GT = self.accuracy["GT"][i] #Retrieves the ground truth answer
            if answer == GT:
                scores.append(1.0) #Appends 1.0 to scores if the answer matches.
            else:
                scores.append(0.0) #Appends 0.0 to scores if the answer does not match.
        scores = sum(scores) / len(scores)
        return scores

    def eval_chatGPT(self, data):
      with ThreadPool(8) as p:
          scores = p.map(self.chatgpt_eval.forward, data)

      # Extract numeric scores from the responses
      numeric_scores = [float(re.findall(r'\d+', score)[0]) for score in scores if re.findall(r'\d+', score)]

      # Calculate the average score
      if numeric_scores:
          average_score = sum(numeric_scores) / len(numeric_scores)
      else:
          average_score = 0.0

      return average_score

    def eval_language(self):
        """
        return the dict evaluation results
        """
        answer = self.language["answer"]
        GT = self.language["GT"]
        results_gen = self.language_eval.run_evaluation(answer, GT)
        # Converts the results to a dictionary format.
        results_gen_dict = {
            f"val/{k}": v for k, v in results_gen.items()
        }
        return results_gen_dict

    def eval_match(self):
        outs1 = []
        for i in range(len(self.match["match"]["answer"])):
            answer = self.match["match"]["answer"][i]
            GT = self.match["match"]["GT"][i]
            _, F1_score = self.match_result(answer, GT)
            outs1.append(F1_score * 100)

        outs1 = sum(outs1) / len(outs1)
        outs2 = self.eval_chatGPT(self.match["GPT"]) # Evaluates the GPT data using ChatGPT.
        scores = (outs1 + outs2) / 2.0
        return scores

    # Defines the method to evaluate if the question is in the graph.
    def eval_graph(self, question):
        # check if answer in self.graph
        question_nums = re.findall(r'\d+\.\d+', question) # Extracts numerical values from the question using regular expressions.
        # Converts the extracted values to a numpy array and reshapes it.
        question_nums = np.array([list(map(float, x.split()))[0] for x in question_nums]).reshape(-1, 2)
        question_nums = [list(i) for i in question_nums] # Converts the numpy array to a list of lists.
        for q in question_nums:
            if q not in self.graph:
                return False
        return True
    # Calculate the match result between the answer and ground truth.
    def match_result(self, answer, GT):
        """
        answer: [[1.,2.], [2., 3.]]
        GT: [[1., 2.], [2., 3.]]
        """
        # Extracting Numerical Values:
        answer_nums = re.findall(r'\d+\.\d+', answer) # Extracts numerical values from the answer.
        GT_nums = re.findall(r'\d+\.\d+', GT) # Extracts numerical values from the ground truth.
        # transform string into float
        if len(answer_nums) % 2 != 0:
            answer_nums = answer_nums[:-1]
        # Converting to Numpy Arrays:
        answer_nums = np.array([list(map(float, x.split()))[0] for x in answer_nums]).reshape(-1, 2)
        GT_nums = np.array([list(map(float, x.split()))[0] for x in GT_nums]).reshape(-1, 2)
        length = len(GT_nums)
        #Initializing Variables for Matching:
        matched_out = []
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        for pred in answer_nums:
            closest_distance = float('inf')
            closest_gt = None
            closest_id = None
            for i, gt in enumerate(GT_nums):
                distance = np.sum(np.abs(pred - gt)) # Calculates the distance between the prediction and ground truth.
                if distance < closest_distance:
                    closest_distance = distance
                    closest_gt = gt
                    closest_id = i

            if closest_distance < 16: # Checks if the closest distance is within a threshold.
                true_positives += 1
                matched_out.append(closest_gt)  # Adds the closest ground truth value to the matched output.
                GT_nums = np.delete(GT_nums, closest_id, axis=0)  # Removes the matched ground truth value from GT_nums.
            else:
                false_positives += 1
        false_negatives = length - true_positives
        precision = true_positives / (true_positives + false_positives + 1e-8)
        recall = true_positives / (true_positives + false_negatives + 1e-8)
        F1 = 2 * precision * recall / (precision + recall + 1e-8)
        return matched_out, F1

    def set_graph(self, answer, GT):
        self.graph, _ = self.match_result(answer, GT)
        self.graph = [list(i) for i in self.graph]

    def forward(self, answer, GT):
        self.GPT = []
        self.accuracy = {"answer": [], "GT": []}
        self.language = {"answer": [], "GT": []}
        self.match = {"match": {"answer": [], "GT": []}, "GPT": []}
        self.accuracy["answer"].append(answer)
        self.accuracy["GT"].append(GT)
        self.GPT.append((answer, GT))
        self.language["GT"].append(GT)
        self.language["answer"].append(answer)
        self.match["match"]["GT"].append(GT)
        self.match["match"]["answer"].append(answer)
        self.match["GPT"].append((answer, GT))


    def evaluation(self):
        print("evaluation start!")
        scores = {}
        scores["accuracy"] = self.eval_acc()
        scores["chatgpt"] = self.eval_chatGPT(self.GPT)
        scores["language"] = self.eval_language()
        scores["match"] = self.eval_match()

        return scores

    # Added optimization function
    # Defines the method to optimize the policy using reinforcement learning.
    def optimize_policy_with_rl(self, data):
        print("Optimization started.")
        for index, item in enumerate(data):
            answer, GT = item
            #Encodes the answer into token IDs.
            input_ids = self.tokenizer.encode(answer, return_tensors='pt')
            rewards = self.reward_model(input_ids) #passes the token IDs through the reward model to get rewards.
            loss = -torch.mean(rewards) # Calculates the loss as the negative mean of the rewards.
            self.optimizer.zero_grad() # Resets the gradients of the optimizer.
            loss.backward() # Computes the gradients of the loss with respect to the model parameters.
            self.optimizer.step() # Updates the model parameters based on the computed gradients.
            print(f"Batch {index + 1}: Loss = {loss.item()}")
        print("Optimization step completed.")

if __name__ == '__main__':
    # get args
    parser = argparse.ArgumentParser(description='Evaluation')
    parser.add_argument('--root_path1', type=str, default="./llama-adapter-DriveLM.json", help='path to prediction file')
    parser.add_argument('--root_path2', type=str, default="./test_v1.json", help='path to test file')
    parser.add_argument('--num_rounds', type=int, default=5, help='number of RLHF rounds')  # Added num_rounds argument
    args = parser.parse_args()
    final_score_list = []
    with open(args.root_path1, 'r') as f:
        pred_file = json.load(f)
    # pred_file = {pred_file[i]["id"]: pred_file[i] for i in range(len(pred_file))}

    with open(args.root_path2, 'r') as f:
        test_file = json.load(f)

    evaluation = evaluation_suit()

    # Running RLHF rounds
    for round_num in range(args.num_rounds): # Added loop for multiple rounds
      print(f"Starting RLHF round {round_num + 1}/{args.num_rounds}")
      for i, qa in enumerate(test_file):
          question = qa["question"]
          GT = qa["answer"]
          idx = qa["id"]
          print(GT)
          predict = next(item for item in pred_file if item["id"] == idx)["answer"]
          print(predict)
          # assert pred_file[idx]["gt_answer"] == GT, print(pred_file[idx]["gt_answer"], GT)
          evaluation.forward(predict, GT)
          print(f"Processed QA pair {i+1}/{len(test_file)}")  # Debugging statement

      output = evaluation.evaluation()
      print("accuracy score: ", output["accuracy"])
      print("chatgpt score: ", output["chatgpt"])
      print("match score: ", output["match"])
      print("language score: ", output["language"])
      # Normalize to 0-1 and combine the scores: chatgpt, language, match, accuracy
      scores = []
      weights = [0.4, 0.2, 0.2, 0.2]
      # chatGPT
      score = output["chatgpt"] / 100.
      scores.append(score)
      # language
      score = 0
      for idx, key in enumerate(output["language"].keys()):
          if idx < 4:
              score += output["language"][key] / 4. / 3.
          elif idx == 4:
              score += output["language"][key] / 3.
          else:
              score += output["language"][key] / 10. / 3.

      scores.append(score)
      # match
      score = output["match"] / 100.
      scores.append(score)
      # accuracy
      score = output["accuracy"]
      scores.append(score)

      final_score = sum([x * y for x, y in zip(scores, weights)])
      print("final score: ", final_score)

      # Optimizes the policy using reinforcement learning based on the GPT data.
      evaluation.optimize_policy_with_rl(evaluation.GPT)  # Added RL optimization step
      print(f"Finished RLHF round {round_num + 1}/{args.num_rounds}")
      final_score_list.append(final_score)
      print("Evaluation suit state:", evaluation.__dict__)
    print(final_score_list)










 #Start

  #Created RewardModel class
    #- Initializes the class
      #- Load a pre-trained GPT-2 model
      #- Add linear layer
    #- Define the forward pass of the model
      #- Pass the input data through the GPT-2 model
      #- Extract the hidden states from the last layer of the GPT-2 model
      #- Compute rewards using the linear layer

  #Initialize Evaluation Suite
    #- Load pre-trained GPT-2 model
    #- Set up language evaluator (BLEU, ROUGE_L, CIDEr)
    #- Initialize reward model and optimizer
    #- Initialize data structures for evaluation metrics (accuracy, language, match, GPT)
    #- Define the method to optimize the policy using reinforcement learning.

  #Load Prediction and Test Files
    #- Load prediction file from `root_path1`
    #- Load test file from `root_path2`
    #- Load the numbers of round

  #RLHF Rounds
    #- Loop for the specified number of rounds (num_rounds)
      #- For each round:
        #- Print starting RLHF round message
        #- For each scene in test file:
          #- For each frame in the scene:
            # - Retrieve QA data
            # - If first QA pair, set the graph
            # - For subsequent QA pairs, evaluate and store metrics if in the graph
        #- Calculate and print evaluation scores (accuracy, chatgpt, match, language)
        #- Normalize and combine scores
        #- Print final score
        #- Optimize policy with RL using GPT data

  #End