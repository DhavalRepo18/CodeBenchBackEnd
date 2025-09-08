import argparse
import json
import os
from collections import defaultdict
from dotenv import load_dotenv
from reactxen.agents.evaluation_agent.agent import EvaluationAgent
from reactxen.utils.model_inference import watsonx_llm
from datasets import load_dataset
from agent_hive.logger import get_custom_logger
from huggingface_hub import login

load_dotenv()
logger = get_custom_logger(__name__)
login(os.getenv("HF_APIKEY", None))


def generate_html_report(
    backstage_directory,
    ouput_directory,
    results_file="evaluation_result.json",
    criteria_distribution_file="criteria_distribution.json",
    output_file="detailed_results.html",
):
    # Load results
    results_file = os.path.join(backstage_directory, results_file)
    with open(results_file, "r") as f:
        results = json.load(f)

    criteria_distribution_file = os.path.join(
        backstage_directory, criteria_distribution_file
    )
    with open(criteria_distribution_file, "r") as f:
        criteria_distribution = json.load(f)

    num_files = len(results)

    # Start HTML
    html = """
    <html>
    <head>
        <title>Detailed Evaluation Results</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1, h2 { color: #333; }
            table { border-collapse: collapse; width: 100%; margin-bottom: 30px; }
            th, td { border: 1px solid #ccc; padding: 8px; text-align: center; }
            th { background-color: #f2f2f2; }
            .true { background-color: #c8e6c9; }   /* green */
            .false { background-color: #ffcdd2; }  /* red */
        </style>
    </head>
    <body>
    """

    # Summary
    html += f"<h1>Detailed Evaluation Results</h1>"
    html += f"<p><b>Number of files processed:</b> {num_files}</p>"

    # Model-level distribution
    html += "<h2>Model-level Criteria Distribution</h2>"
    for model_id, criteria in criteria_distribution.items():
        html += f"<h3>Model {model_id}</h3>"
        html += "<table><tr><th>Criterion</th><th>True</th><th>False</th></tr>"
        for criterion, counts in criteria.items():
            html += f"<tr><td>{criterion}</td><td>{counts['True']}</td><td>{counts['False']}</td></tr>"
        html += "</table>"

    # Question-level results (no leakage of sensitive fields)
    html += "<h2>Question-level Results</h2>"
    html += "<table><tr><th>#</th>"
    # Dynamically detect criteria fields
    criteria_fields = [
        k
        for k in results[0].keys()
        if k not in ["suggestions", "question", "model_id", "question_id"]
    ]
    for c in criteria_fields:
        html += f"<th>{c}</th>"
    html += "</tr>"

    for i, result in enumerate(results, 1):
        html += f"<tr><td>{i}</td>"
        for c in criteria_fields:
            val = result[c]
            css_class = "true" if val else "false"
            html += f"<td class='{css_class}'>{val}</td>"
        html += "</tr>"
    html += "</tr></table>"

    # End HTML
    html += "</body></html>"

    output_file = os.path.join(ouput_directory, output_file)
    os.makedirs(
            os.path.dirname(output_file), exist_ok=True
        )
    with open(output_file, "w") as f:
        f.write(html)

    print(f"✅ HTML report written to {output_file}")


def load_scenarios(utterance_ids):
    ds = load_dataset("ibm-research/AssetOpsBench", "scenarios")
    train_ds = ds["train"]
    df = train_ds.to_pandas()
    filtered_df = df[df["id"].isin(utterance_ids)]
    return filtered_df.to_dict(orient="records")


def evaluate(
    dict_of_utterances,
    traj_directory="./localtemp/trajectory/",
    backstage_directory="./",
):
    all_results = []

    for file in os.listdir(traj_directory):
        # check if the file is a json file
        if file.endswith(".json"):
            splits = file.split("_")
            model_id = 12
            question_id = int(splits[1])
            # open the file
            with open(os.path.join(traj_directory, file), "r") as f:
                # read the json data
                # print('reading file', file)
                data = json.load(f)
                task = data["text"]
                final_answer = data["trajectory"][-1]["response"]

                trajectory = data["trajectory"]
                agent_think = "The agent executes the following steps: "
                for item in trajectory:
                    agent_think += f"{item['task_number']}. task: {item['task_description']}; agent: {item['agent_name']}; response: {item['response']}. "
                utterance = dict_of_utterances[question_id]
                assert (
                    utterance["text"] == task
                ), f"task mismatch: {utterance['text']} vs {task}"
                evaluation_agent = EvaluationAgent(
                    llm=watsonx_llm, model_id="mistralai/mistral-large", max_retries=2
                )
                result = evaluation_agent.evaluate_response(
                    question=utterance["text"],
                    agent_response=final_answer,
                    characteristic_answer=utterance["characteristic_form"],
                    agent_think=agent_think,
                )
                result["question"] = utterance["text"]
                result["model_id"] = model_id
                result["question_id"] = question_id
                all_results.append(result)

    results_file = os.path.join(backstage_directory, "evaluation_result.json")
    os.makedirs(
            os.path.dirname(results_file), exist_ok=True
        )

    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=4)

    criteria_distribution_by_model = defaultdict(dict)

    # Process each result and organize the criteria
    for result in all_results:
        model_id = result["model_id"]

        for criterion, value in result.items():
            if criterion not in ["question", "model_id", "question_id", "suggestions"]:
                if criterion not in criteria_distribution_by_model[model_id]:
                    criteria_distribution_by_model[model_id][criterion] = {
                        "True": 0,
                        "False": 0,
                    }

                criteria_distribution_by_model[model_id][criterion][
                    str(value).capitalize()
                ] += 1

    # Save the distribution to a file
    results_file = os.path.join(backstage_directory, "criteria_distribution.json")
    with open(results_file, "w") as f:
        json.dump(criteria_distribution_by_model, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--utterance_ids", type=str, default="1,106")
    parser.add_argument("--traj_directory", type=str, default="./localtemp/trajectory/")
    parser.add_argument("--backstage_directory", type=str, default=".")
    parser.add_argument("--ouput_directory", type=str, default=".")

    args = parser.parse_args()
    traj_directory = args.traj_directory
    utterance_ids = [int(uid.strip()) for uid in args.utterance_ids.split(",")]
    utterances = load_scenarios(utterance_ids)
    dict_of_utterances = {int(item["id"]): item for item in utterances}
    evaluate(dict_of_utterances, traj_directory, args.backstage_directory)
    generate_html_report(args.backstage_directory, args.ouput_directory)
