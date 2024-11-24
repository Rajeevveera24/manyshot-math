import asyncio
import json
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import aiofiles

# Assuming logging setup is done above


class ExperimentLogger:
    """A class for logging and managing experiment results.

    This class provides functionality to:
    - Store experiment results in memory (e.g. model outputs, accuracy scores)
    - Automatically save results to a JSON file when the buffer exceeds log_frequency entries
    - Generate summary statistics of experiment outcomes (total questions, accuracy, etc.)
    - Force manual saving (ie writing) of results to file
    - Thread-safe operations for concurrent access
    - Asynchronous background saving to disk

    The class uses thread-safe operations to handle concurrent access to results.
    Results are periodically saved to disk in a background task to prevent data loss.

    The results file will be saved in the ../experiments directory with a timestamped
    unique identifier if no filename is provided.
    """

    def __init__(self, results_file: Optional[str] = None, logging_frequency: int = 20, add_timestamp=False):
        """Initialize the ExperimentLogger.

        Args:
            results_file (str, optional): Path to the JSON file where results will be saved.
                                        Defaults to "experiment_results.json".
        """
        self.results = []
        self.logging_frequency = logging_frequency
        if results_file is None:
            # Generate unique filename with timestamp
            timestamp = datetime.now().strftime("%m%d%H")
            unique_id = str(uuid.uuid4())[:8]
            filename = f"experiment_{timestamp}_{unique_id}.json"

            # Create experiments directory if it doesn't exist
            exp_dir = Path("../experiments")
            exp_dir.mkdir(exist_ok=True)

            # Set results file path
            self.results_file = str(exp_dir / filename)
            print(f"Results will be saved to: {self.results_file}")
        else:
            if add_timestamp:
                timestamp = datetime.now().strftime("%m%d%H")
                self.results_file = results_file + "_" + timestamp
                print(f"Results will be saved to: {self.results_file}")
            self.results_file = results_file

        self._results_lock = threading.Lock()  # Create lock for results access
        # Start the background task
        # loop = asyncio.get_event_loop()
        # if not loop.is_running():
        #     loop = asyncio.new_event_loop()
        #     asyncio.set_event_loop(loop)
        # loop.create_task(self._monitor_and_save_results())

    def get_results(self):
        """Return all logged results."""
        return self.results

    def get_summary(self):
        """Get summary statistics of the experiment."""
        total = len(self.results)
        correct = sum(1 for r in self.results if r["correct"])
        return {
            "total_questions": total,
            "correct_answers": correct,
            "accuracy": correct / total if total > 0 else 0,
        }

    # async def _monitor_and_save_results(self):
    #     """Background task that monitors results length and saves to file."""

    #     while True:
    #         if len(self.results) >= self.logging_frequency:
    #             try:
    #                 with self._results_lock:  # Acquire lock before accessing results
    #                     results_to_save = (
    #                         self.results.copy()
    #                     )  # Make a copy while locked
    #                     self.results = []  # Clear original under lock

    #                 async with aiofiles.open(self.results_file, mode="a") as f:
    #                     await f.write(json.dumps(results_to_save, indent=2))
    #                     await f.write("\n")  # Add newline between batches

    #                 print(f"Successfully saved results batch to {self.results_file}")
    #             except Exception as e:
    #                 print(f"Error saving results: {str(e)}")
    #         await asyncio.sleep(0.2)  # Check every second

    async def force_save_results(self):
        """Force save all current results to file."""
        try:
            with self._results_lock:  # Acquire lock before accessing results
                results_to_save = self.results.copy()  # Make a copy while locked
                self.results = []  # Clear results after copying
            
            print(f"Force saving {len(results_to_save)} results to {self.results_file}")
            async with aiofiles.open(self.results_file, mode="a") as f:
                await f.write(json.dumps(results_to_save, indent=2))
                await f.write("\n")  # Add newline between batches

            print(f"Successfully force saved results to {self.results_file}")
        except Exception as e:
            print(f"Error force saving results: {str(e)}")
            # Restore results if save failed
            with self._results_lock:
                self.results.extend(results_to_save)

    def log_result(self, question_id, predicted_answer, true_answer=None):
        """Log a single result to the experiment results."""
        with self._results_lock:  # Acquire lock before modifying results
            result = {
                "question_id": question_id,
                "predicted_answer": predicted_answer,
            }
            self.results.append(result)

    def log_results(self, results_list):
        """Log multiple results at once.

        Args:
            results_list: List of dictionaries, each containing:
                - question_id: ID of the question
                - predicted_answer: The predicted answer
                - true_answer: Optional true answer
        """
        with self._results_lock:  # Acquire lock before modifying results
            for result_data in results_list:
                result = {
                    "question_id": result_data["question_id"],
                    "predicted_answer": result_data["predicted_answer"],
                }
                self.results.append(result)

    def write_results_to_file(self):
        """Write all results to file synchronously by wrapping the async force_save_results."""
        asyncio.run(self.force_save_results())
        print(f"Results written to {self.results_file}")

    def __str__(self):
        return f"ExperimentLogger logging to: {self.results_file}"


## Example usage:
# if __name__ == "__main__":
#     import asyncio

#     async def main():
#         # Create logger instance
#         logger = ExperimentLogger()

#         # Log some example results
#         logger.log_result(
#             question_id="q1",
#             predicted_answer="The capital of France is Paris",
#             true_answer="Paris is the capital of France",
#         )

#         logger.log_result(
#             question_id="q2",
#             predicted_answer="The sky is blue",
#             true_answer="The sky appears blue due to Rayleigh scattering",
#         )

#         # Get summary statistics
#         summary = logger.get_summary()
#         print("Experiment Summary:")
#         print(f"Total Questions: {summary['total_questions']}")
#         print(f"Correct Answers: {summary['correct_answers']}")
#         print(f"Accuracy: {summary['accuracy']:.2%}")

#         # Get all results
#         all_results = logger.get_results()
#         print("\nAll Results:")
#         for result in all_results:
#             print(result)

#         # Keep the program running to allow background task to work
#         await asyncio.sleep(5)

#     # Run the example
#     asyncio.run(main())
