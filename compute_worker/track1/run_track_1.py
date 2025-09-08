import argparse
import json
import os
import sys
import importlib.util
import warnings

from dotenv import load_dotenv
from datasets import load_dataset
from huggingface_hub import login

from agent_hive.task import Task
from agent_hive.tools.fmsr import (
    fmsr_tools,
    fmsr_fewshots,
    fmsr_task_examples,
    fmsr_agent_name,
    fmsr_agent_description,
)
from agent_hive.tools.skyspark import (
    iot_bms_tools,
    iot_bms_fewshots,
    iot_agent_description,
    iot_agent_name,
)
from agent_hive.tools.tsfm import (
    tsfm_tools,
    tsfm_fewshots,
    tsfm_agent_name,
    tsfm_agent_description,
)
from agent_hive.tools.wo import (
    wo_agent_description,
    wo_agent_name,
    wo_fewshots,
    wo_tools,
)
from agent_hive.agents.react_agent import ReactAgent
from agent_hive.logger import get_custom_logger
from agent_hive.agents.wo_agent import WorderOrderAgent

logger = get_custom_logger(__name__)

load_dotenv()
login(os.getenv("HF_APIKEY", None))

warnings.filterwarnings("ignore")


def load_dynamic_workflow(file_path: str):
    """
    Dynamically import NewPlanningWorkflow from a given Python file path.
    """
    module_name = os.path.splitext(os.path.basename(file_path))[
        0
    ]  # e.g. "track1_planning"
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    return getattr(module, "NewPlanningWorkflow")


def load_scenarios(utterance_ids):
    ds = load_dataset("ibm-research/AssetOpsBench", "scenarios")
    train_ds = ds["train"]
    df = train_ds.to_pandas()
    filtered_df = df[df["id"].isin(utterance_ids)]
    return filtered_df.to_dict(orient="records")


def run_planning_workflow(
    NewPlanningWorkflow, question, qid, llm_model=16, generate_steps_only=False
):
    iot_r_agent = ReactAgent(
        name=iot_agent_name,
        description=iot_agent_description,
        tools=iot_bms_tools,
        llm=llm_model,
        few_shots=iot_bms_fewshots,
    )

    fmsr_r_agent = ReactAgent(
        name=fmsr_agent_name,
        description=fmsr_agent_description,
        tools=fmsr_tools,
        llm=llm_model,
        task_examples=fmsr_task_examples,
        few_shots=fmsr_fewshots,
    )

    tsfm_rr_agent = ReactAgent(
        name=tsfm_agent_name,
        description=tsfm_agent_description,
        tools=tsfm_tools,
        llm=llm_model,
        few_shots=tsfm_fewshots,
        reflect_step=1
    )
    
    wo_rr_agent = WorderOrderAgent(
        name=wo_agent_name,
        description=wo_agent_description,
        tools=wo_tools,
        llm=llm_model,
        few_shots=wo_fewshots,
        reflect_step=1
    )


    task = Task(
        description=question,
        expected_output="",
        agents=[iot_r_agent, fmsr_r_agent, tsfm_rr_agent, wo_rr_agent],
    )

    wf = NewPlanningWorkflow(
        tasks=[task],
        llm=llm_model,
    )

    return wf.run()


def run(
    NewPlanningWorkflow,
    utterances,
    generate_steps_only=False,
    traj_directory="./localtemp/trajectory/",
):

    os.makedirs(traj_directory, exist_ok=True)

    for utterance in utterances:
        logger.info("=" * 10)
        logger.info(f"ID: {utterance['id']}, Task: {utterance['text']}")

        trajectory_file = os.path.join(
            traj_directory, f"Q_{utterance['id']}_trajectory.json"
        )

        ans = run_planning_workflow(
            NewPlanningWorkflow,
            utterance["text"],
            utterance["id"],
            generate_steps_only=generate_steps_only,
        )

        if generate_steps_only:
            continue

        output = {"id": utterance["id"], "text": utterance["text"], "trajectory": ans}

        with open(trajectory_file, "w") as f:
            json.dump(output, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--utterance_ids", type=str, default="1,106")
    parser.add_argument("--generate_steps_only", type=bool, default=False)
    parser.add_argument(
        "--workflow_path",
        type=str,
        default=None,
        help="Path to custom NewPlanningWorkflow.py",
    )
    parser.add_argument(
        "--traj_directory",
        type=str,
        default=None,
        help="Path to store all output trajectroy",
    )

    args = parser.parse_args()
    utterance_ids = [int(uid.strip()) for uid in args.utterance_ids.split(",")]
    utterances = load_scenarios(utterance_ids)
    traj_directory = args.traj_directory

    # load workflow dynamically if path given, otherwise default
    if args.workflow_path:
        NewPlanningWorkflow = load_dynamic_workflow(args.workflow_path)
        logger.info(f"✅ Loaded NewPlanningWorkflow from {args.workflow_path}")
    else:
        from track1_planning import NewPlanningWorkflow

        logger.info("✅ Loaded default NewPlanningWorkflow")

    run(
        NewPlanningWorkflow,
        utterances,
        generate_steps_only=args.generate_steps_only,
        traj_directory=traj_directory,
    )
