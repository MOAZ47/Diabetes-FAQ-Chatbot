# wnb_eval.py

import wandb
import datetime
import pandas as pd
import traceback
import os, json

# Import your evaluation setup
import evaluation

def run_wandb_logged_evaluation():
    # 🪪 Start a W&B run
    run = wandb.init(
        project="diabetes-qa-evaluation",
        name=f"eval-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}",
        config={
            "model": evaluation.llm_provider.model_engine,
            "temperature": evaluation.TEMP,
            "app_name": evaluation.APP_NAME,
            "app_version": evaluation.APP_VERSION,
        },
    )

    try:
        # Run your evaluation
        evaluation.run_evaluation()

        #queries = evaluation.test_queries
        # Save test queries to a file
        queries_path = "./logs/test_queries.jsonl"
        os.makedirs(os.path.dirname(queries_path), exist_ok=True)
        with open(queries_path, "w", encoding="utf-8") as f:
            for q in evaluation.test_queries:
                f.write(json.dumps({"query": q}) + "\n")
        
        # Log leaderboard
        leaderboard_df = evaluation.tru.get_leaderboard(app_ids=[evaluation.tru_app.app_id])
        wandb.log({"leaderboard": wandb.Table(dataframe=leaderboard_df)})

        # Log feedbacks
        records_df, feedback_df = evaluation.tru.get_records_and_feedback(app_ids=[evaluation.tru_app.app_id])

        wandb.log({
            "total_records": len(records_df),
            "total_feedbacks": len(feedback_df),
            "records": wandb.Table(dataframe=records_df)
        })

        # save leaderboard as artifact
        leaderboard_artifact = wandb.Artifact("leaderboard", type="eval")
        leaderboard_artifact.add_file("./logs/trulens_eval_results.json")
        run.log_artifact(leaderboard_artifact)

        # 2. Log queries as W&B artifact
        queries_artifact = wandb.Artifact("test_queries", type="dataset")
        queries_artifact.add_file(queries_path)
        run.log_artifact(queries_artifact)

        print("✅ Logged to W&B successfully.")

    except Exception as e:
        print("❌ Evaluation failed:", e)
        traceback.print_exc()
        wandb.alert(
            title="TruLens Evaluation Failed",
            text=f"{type(e).__name__}: {str(e)}"
        )
    finally:
        run.finish()


if __name__ == "__main__":
    run_wandb_logged_evaluation()
