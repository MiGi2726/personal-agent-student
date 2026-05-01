from __future__ import annotations

from typing import Any

from course_project.json_tools import (
    exhausted_invalid_feedback,
    invalid_json_feedback,
    parse_action,
)
from course_project.student_api import SessionResult, StudentAgent, StudentRuntime

from .common import assistant_message, build_system_prompt, tool_result_message


class BaselineStudentAgent(StudentAgent):
    def __init__(self, settings: Any) -> None:
        self.settings = settings

    def run_session(self, session, runtime: StudentRuntime) -> SessionResult:  # noqa: ANN001
        """Run one Stage-1 scheduling session.

        This is the same four-step loop from mini-stages 1–5, assembled into
        a working agent. The loop calls the model, parses the response, runs
        the requested tool, and repeats until the model returns a final answer.

        Design choices are yours:
        - What rules go in prompts.py? (Run mini-stage 6 to see the baseline first.)
        - How many invalid-JSON retries before giving up?
        - When is the agent truly done — final_response alone, or verified event creation?
        """
        messages: list[dict[str, Any]] = [{"role": "user", "content": session.user_message}]
        invalid_response_count = 0
        final_response = ""

        for turn_index in range(runtime.max_model_turns):
            system_prompt = build_system_prompt(runtime)

            # STEP 1: call the model (same as mini_1)
            response = runtime.complete(
                messages=[{"role": "system", "content": system_prompt}] + messages,
                require_json=True,
            )

            # STEP 2: record the response and parse the action (same as mini_2)
            messages.append(assistant_message(response))
            try: 
                action = parse_action(response.content)
            except Exception:
                invalid_response_count += 1
                messages.append({"role": "user", "content": invalid_json_feedback()})
                continue

            # STEP 3: if the model requested a tool, run it and send the result back (same as mini_3 + mini_4)
            tool_call = action.get("tool_call")
            if tool_call:
                tool_name, _, result = runtime.call_tool(
                    tool_name=tool_call.get("name"),
                    arguments=tool_call.get("arguments", {}),
                    turn_index=turn_index,
                )
                messages.append(tool_result_message(tool_name, result))
                # After appending the result, `continue` sends the loop back to STEP 1.
                # On the next iteration the model will see the tool result and ask for the next step.
                # This is how multi-turn works — the loop replaces the manual second call in mini_4.
                continue

            # STEP 4: if the model returned a final response, stop the loop.
            # The model signals it is done by setting final_response to a non-empty string.
            # Check action['final_response']. If it is non-empty, store it in `final_response` and break.
            # TODO: replace the `pass` below with your termination condition (~2 lines).
            if(action["final_response"]):  
                final_response = action["final_response"]
                break


        # Return the result to the benchmark.
        if not final_response and invalid_response_count > 0:
            final_response = exhausted_invalid_feedback()
        if not final_response:
            final_response = (
                "Stage 1 agent is not implemented yet. Complete the TODOs in "
                "student_scaffold/agent.py, then rerun the benchmark."
            )
        return runtime.finish(final_response)


def build_agent(settings: Any) -> BaselineStudentAgent:
    return BaselineStudentAgent(settings)
