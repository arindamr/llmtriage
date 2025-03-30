# --- Imports ---
import os
import logging
import time
import uuid
from typing import List, Dict, Any, Optional, Type
from pydantic import BaseModel, Field
from abc import ABC, abstractmethod
import json

# LangChain specific imports
#from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate, MessagesPlaceholder  # Updated import path for langchain-core==0.3.49
from langchain.tools import BaseTool, tool  # StructuredTool is not required for this version
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage
from langchain.callbacks.base import BaseCallbackHandler

# For environment variables (like API keys)
from dotenv import load_dotenv

# FastAPI imports
from fastapi import FastAPI, HTTPException, Path, Body, status
import uvicorn

# --- Configuration ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Error Handling ---
class ToolExecutionError(Exception):
    """Custom exception for errors during tool execution."""
    pass

# --- State Management ---
class DiagnosticState:
    """Manages the state of a single diagnostic session."""
    # (Keep the DiagnosticState class exactly as defined in the previous example)
    def __init__(self, initial_problem: str):
        self.initial_problem: str = initial_problem
        self.history: List[Dict[str, Any]] = [{"role": "user", "content": initial_problem}]
        self.steps: List[Dict[str, Any]] = []
        self.final_diagnosis: Optional[str] = None
        logging.info(f"Initialized diagnostic state for problem: {initial_problem[:100]}...")

    def add_step(self, step_type: str, data: Dict[str, Any]):
        """Adds a step to the diagnostic process log."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        step_data = {"timestamp": timestamp, "type": step_type, **data}
        self.steps.append(step_data)
        logging.info(f"State Step Added: {step_type} - {data.get('tool', data.get('message', 'N/A'))}")

        # Also add relevant info to Langchain message history
        if step_type == "llm_thought":
             pass # Usually captured within the AIMessage that follows
        elif step_type == "tool_call":
             pass # Represented by AIMessage's tool_calls attribute
        elif step_type == "tool_result":
            # Check if tool_call_id is present and valid
            if "tool_call_id" in data and data["tool_call_id"]:
                self.history.append(ToolMessage(
                    tool_call_id=data["tool_call_id"],
                    content=str(data["output"]) # Ensure content is string
                ))
            else:
                logging.warning("Tool result step missing valid tool_call_id. Not adding to history.")
                # Optionally, add as a system message or observation if needed.
                # self.history.append(SystemMessage(content=f"Observation from tool {data.get('tool', 'unknown')}: {data['output']}"))

        elif step_type == "error":
             # Check if tool_call_id is present and valid
             if "tool_call_id" in data and data["tool_call_id"]:
                self.history.append(ToolMessage(
                    tool_call_id=data["tool_call_id"],
                    content=f"Error executing tool {data['tool']}: {data['error']}"
                ))
             else:
                 logging.warning("Tool error step missing valid tool_call_id. Not adding error to history.")
                 # Optionally add error info differently if needed
        elif step_type == "final_diagnosis":
            self.final_diagnosis = data["diagnosis"]

    def add_user_input(self, user_input: str):
        """Adds additional user input to the diagnostic process."""
        self.history.append({"role": "user", "content": user_input})
        logging.info(f"Added user input to diagnostic state: {user_input[:100]}...")

    def get_state_summary(self) -> Dict[str, Any]:
        """Returns a summary of the current diagnostic state."""
        # Ensure history is serializable (convert BaseMessage objects if needed)
        serializable_history = []
        for msg in self.get_langchain_history():
            if isinstance(msg, HumanMessage):
                serializable_history.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                entry = {"role": "assistant", "content": msg.content}
                if msg.tool_calls:
                    entry["tool_calls"] = msg.tool_calls
                serializable_history.append(entry)
            elif isinstance(msg, ToolMessage):
                 serializable_history.append({"role": "tool", "content": msg.content, "tool_call_id": msg.tool_call_id})
            # Add other message types if used (SystemMessage, etc.)

        return {
            "initial_problem": self.initial_problem,
            "steps_taken": self.steps,
            "final_diagnosis": self.final_diagnosis,
            "conversation_history": serializable_history # Use serializable history
        }

    def get_langchain_history(self) -> List[BaseMessage]:
        """Formats history for Langchain agent input"""
        processed_history = []
        # Check if self.history contains dicts or BaseMessages and convert/use directly
        temp_history_to_process = list(self.history) # Work on a copy

        for item in temp_history_to_process:
            if isinstance(item, BaseMessage):
                processed_history.append(item)
            elif isinstance(item, dict):
                role = item.get("role")
                content = item.get("content", "") # Ensure content is not None
                tool_calls = item.get("tool_calls")
                tool_call_id = item.get("tool_call_id")

                if role == "user":
                    processed_history.append(HumanMessage(content=str(content)))
                elif role == "assistant" or role == "ai":
                    if tool_calls:
                        processed_history.append(AIMessage(content=str(content), tool_calls=tool_calls))
                    else:
                        processed_history.append(AIMessage(content=str(content)))
                elif role == "tool":
                    if tool_call_id: # Only add ToolMessage if tool_call_id is present
                         processed_history.append(ToolMessage(content=str(content), tool_call_id=tool_call_id))
                    else:
                         logging.warning(f"Skipping ToolMessage creation due to missing tool_call_id: {item}")

        # Update self.history to be the BaseMessage list for future Langchain calls
        # Note: This assumes subsequent calls within the *same* DiagnosticState instance
        # If state is persisted and reloaded, ensure reloading reconstructs BaseMessages
        # self.history = processed_history # Be cautious with this line if state is serialized/deserialized simply as dicts

        return processed_history


# --- Tool Definition and Registry ---
# (Keep CheckServiceInput, ViewLogsInput, TestDatabaseInput Pydantic models as defined before)
class CheckServiceInput(BaseModel):
    service_name: str = Field(description="The name of the service to check, e.g., 'auth-service', 'payment-gateway'.")
    environment: str = Field(description="The environment to check the service in, e.g., 'production', 'staging'.")

class ViewLogsInput(BaseModel):
    service_name: str = Field(description="The name of the service whose logs should be viewed.")
    environment: str = Field(description="The environment where the service is running.")
    lines: int = Field(default=50, description="Number of recent log lines to retrieve.")
    filter_keyword: Optional[str] = Field(default=None, description="Optional keyword to filter logs.")

class TestDatabaseInput(BaseModel):
    database_name: str = Field(description="The name of the database to test.")
    query: Optional[str] = Field(default="SELECT 1;", description="A simple query to test database connectivity and responsiveness.")


# --- Placeholder Tool Implementations ---
# (Keep the @tool decorated functions check_service_status, view_recent_logs, test_database_connection as defined before)
@tool("check_service_status", args_schema=CheckServiceInput)
def check_service_status(service_name: str, environment: str) -> str:
    """Checks the operational status of a specified service in a given environment."""
    logging.info(f"Executing check_service_status: service={service_name}, env={environment}")
    # --- Dummy Implementation ---
    if service_name == "auth-service" and environment == "production":
        return json.dumps({"status": "running", "version": "1.2.3", "instances": 3})
    elif service_name == "payment-gateway":
        if time.time() % 10 < 5:
             raise ToolExecutionError(f"Failed to connect to {service_name} monitoring endpoint in {environment}.")
        return json.dumps({"status": "degraded", "version": "2.1.0", "instances": 2, "error_rate": "5%"})
    else:
        return json.dumps({"status": "unknown", "message": f"Service {service_name} not found in {environment}."})

@tool("view_recent_logs", args_schema=ViewLogsInput)
def view_recent_logs(service_name: str, environment: str, lines: int = 50, filter_keyword: Optional[str] = None) -> str:
    """Views recent log entries for a specified service, with optional filtering."""
    logging.info(f"Executing view_recent_logs: service={service_name}, env={environment}, lines={lines}, filter={filter_keyword}")
    # --- Dummy Implementation ---
    log_lines = [
        f"INFO: Request received on /api/users",
        f"WARN: Deprecated feature used in request X",
        f"ERROR: Database connection timeout for user Y",
        f"INFO: Request processed successfully",
    ] * (lines // 4)
    if filter_keyword:
        log_lines = [line for line in log_lines if filter_keyword.lower() in line.lower()]
    if service_name == "auth-service" and "error" in (filter_keyword or "").lower():
         log_lines.append("CRITICAL: Failed login attempt limit exceeded for IP 1.2.3.4")
    return "\n".join(log_lines[-lines:])

@tool("test_database_connection", args_schema=TestDatabaseInput)
def test_database_connection(database_name: str, query: str = "SELECT 1;") -> str:
    """Tests connectivity and basic query execution against a specified database."""
    logging.info(f"Executing test_database_connection: db={database_name}, query='{query}'")
    # --- Dummy Implementation ---
    if "orders_db" in database_name:
        if query == "SELECT 1;":
            return json.dumps({"status": "connected", "query_result": 1})
        else:
            return json.dumps({"status": "connected", "query_result": "Query executed, dummy result."})
    else:
         raise ToolExecutionError(f"Failed to connect to database: {database_name}. Check credentials or network.")


# --- Tool Registry ---
# (Keep the ToolRegistry class exactly as defined before)
class ToolRegistry:
    """Holds and provides access to available diagnostic tools."""
    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
        self.register_tool(check_service_status)
        self.register_tool(view_recent_logs)
        self.register_tool(test_database_connection)
        logging.info(f"Tool Registry initialized with tools: {list(self._tools.keys())}")

    def register_tool(self, tool_instance: BaseTool):
        """Registers a single tool."""
        if not isinstance(tool_instance, BaseTool):
            raise TypeError("Provided item is not a valid LangChain BaseTool")
        if tool_instance.name in self._tools:
            logging.warning(f"Tool '{tool_instance.name}' is already registered. Overwriting.")
        self._tools[tool_instance.name] = tool_instance

    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Retrieves a tool by its name."""
        return self._tools.get(name)

    def get_all_tools(self) -> List[BaseTool]:
        """Returns a list of all registered tools."""
        return list(self._tools.values())

    def get_tool_definitions_for_llm(self) -> List[Dict[str, Any]]:
         """Returns tool definitions in the format expected by OpenAI function calling."""
         return [convert_to_openai_tool(tool) for tool in self._tools.values()]


# --- LLM Core: The Triage Agent ---
# (Keep the TriageSystem class mostly as defined before, but adjust run_triage slightly)
class TriageSystem:
    """Orchestrates the automated triage process using an LLM agent."""
    def __init__(self, tool_registry: ToolRegistry):
        self.tool_registry = tool_registry
        # Ensure OPENAI_API_KEY is set
        if not os.getenv("OPENAI_API_KEY"):
             raise ValueError("OPENAI_API_KEY environment variable not set.")
        self.llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)  # Updated model name for compatibility

        prompt_template = ChatPromptTemplate.from_messages([
             ("system", """You are an expert AI assistant for automated software issue triage... [Your detailed system prompt here, same as before]"""), # Keep the detailed prompt
             MessagesPlaceholder(variable_name="chat_history"),
             ("human", "{input}"),
             MessagesPlaceholder(variable_name="agent_scratchpad")
        ])

        tools = self.tool_registry.get_all_tools()
        self.agent = create_openai_tools_agent(self.llm, tools, prompt_template)

        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=tools,
            verbose=True, # Good for debugging API calls
            handle_parsing_errors=self._handle_agent_parsing_error, # Custom handler for API context
            max_iterations=10
        )
        logging.info("TriageSystem initialized with LLM and Agent Executor.")

    # Custom parsing error handler to provide more informative API errors
    def _handle_agent_parsing_error(self, error: Exception) -> str:
        error_str = str(error)
        logging.error(f"Agent parsing error: {error_str}")
        # You might want to extract specific details if possible
        # Return a message instructing the agent how to recover or inform the user
        return f"Could not parse the LLM's response. Error: {error_str}. Please try again or rephrase your thinking."

    # Modify run_triage to store the DiagnosticState object directly
    def run_triage(self, problem_statement: str, session_id: str) -> DiagnosticState:
        """Runs the full triage process, returning the final DiagnosticState."""
        diagnostic_state = DiagnosticState(problem_statement)
        logging.info(f"[Session: {session_id}] Starting triage for: {problem_statement[:100]}...")

        # Store the DiagnosticState object directly in session_states
        session_states[session_id] = diagnostic_state

        # --- Langchain Agent Callback Handler ---
        # To capture intermediate steps correctly into DiagnosticState, we use callbacks.
        class StateUpdateCallback(BaseCallbackHandler):
            def __init__(self, state: DiagnosticState):
                self.state = state

            def on_agent_action(self, action: Any, **kwargs: Any) -> Any:
                """Track tool calls."""
                # action is typically AgentAction(tool=..., tool_input=..., log=...)
                log_entry = getattr(action, 'log', '').split('\n')[0] # Get first line of thought
                self.state.add_step("llm_thought", {"message": log_entry})
                # Note: action.tool_call_id isn't directly available here in standard AgentAction
                # We rely on on_tool_start to capture the call details accurately

            def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs: Any) -> Any:
                 """Log tool execution start"""
                 tool_name = serialized.get("name", "unknown_tool")
                 tool_call_id = kwargs.get('run_id') # Or find the actual call ID if passed differently
                 # Parse input_str safely if it's JSON, otherwise use as string
                 try:
                    tool_input = json.loads(input_str)
                 except json.JSONDecodeError:
                    tool_input = input_str # Keep as string if not JSON

                 self.state.add_step("tool_call", {
                     "tool": tool_name,
                     "input": tool_input,
                     "tool_call_id": str(tool_call_id) # Ensure it's a string for ToolMessage later
                 })
                 # Store the call_id mapping if needed (e.g., in self.state or a temp dict)
                 # This ID needs to match what on_tool_end receives

            def on_tool_end(self, output: str, **kwargs: Any) -> Any:
                """Track tool results."""
                # Find the corresponding tool_call_id for this result
                tool_call_id = str(kwargs.get('run_id')) # Should match the ID from on_tool_start
                # Find the step matching this call_id to get the tool name
                calling_step = next((step for step in reversed(self.state.steps) if step.get("tool_call_id") == tool_call_id), None)
                tool_name = calling_step['tool'] if calling_step else 'unknown_tool'

                self.state.add_step("tool_result", {
                    "tool": tool_name,
                    "output": output,
                    "tool_call_id": tool_call_id
                })

            def on_tool_error(self, error: BaseException, **kwargs: Any) -> Any:
                 """Track tool errors."""
                 tool_call_id = str(kwargs.get('run_id'))
                 calling_step = next((step for step in reversed(self.state.steps) if step.get("tool_call_id") == tool_call_id), None)
                 tool_name = calling_step['tool'] if calling_step else 'unknown_tool'
                 self.state.add_step("error", {
                     "tool": tool_name,
                     "error": str(error),
                     "tool_call_id": tool_call_id
                 })

            # We might need on_llm_end or on_agent_finish to capture the final diagnosis reasoning
            def on_agent_finish(self, finish: Any, **kwargs: Any) -> Any:
                """Capture final output."""
                final_output = finish.return_values.get("output", "Agent finished without explicit output.")
                self.state.add_step("final_diagnosis", {"diagnosis": final_output})

        # --- Run Agent ---
        state_callback = StateUpdateCallback(diagnostic_state)
        agent_input = {
            "input": problem_statement,
            "chat_history": diagnostic_state.get_langchain_history()
        }

        try:
            # Run the agent executor with the callback
            response = self.agent_executor.invoke(
                 agent_input,
                 config={"callbacks": [state_callback]}
             )

            # Final output should have been captured by on_agent_finish callback
            if diagnostic_state.final_diagnosis is None:
                 # Fallback if on_agent_finish didn't capture it
                 final_output = response.get("output", "Agent did not provide a final output.")
                 diagnostic_state.add_step("final_diagnosis", {"diagnosis": final_output})

            logging.info(f"[Session: {session_id}] Triage completed.")

        except Exception as e:
            error_message = f"An unexpected error occurred during triage: {e}"
            logging.error(f"[Session: {session_id}] {error_message}", exc_info=True)
            diagnostic_state.add_step("system_error", {"error": error_message})
            diagnostic_state.final_diagnosis = f"Triage failed due to system error: {e}"

        return diagnostic_state # Return the full state object

    def continue_triage(self, session_id: str, additional_input: str) -> DiagnosticState:
        """Continues the triage process with additional user input."""
        # Retrieve the DiagnosticState object directly
        diagnostic_state = session_states.get(session_id)
        if not diagnostic_state:
            raise ValueError("Session ID not found.")

        diagnostic_state.add_user_input(additional_input)

        # Re-run the agent with the updated state
        agent_input = {
            "input": additional_input,
            "chat_history": diagnostic_state.get_langchain_history()
        }

        try:
            response = self.agent_executor.invoke(agent_input)
            final_output = response.get("output", "Agent did not provide a final output.")
            diagnostic_state.add_step("final_diagnosis", {"diagnosis": final_output})
            logging.info(f"[Session: {session_id}] Continued triage completed.")
        except Exception as e:
            error_message = f"An error occurred during continued triage: {e}"
            logging.error(f"[Session: {session_id}] {error_message}", exc_info=True)
            diagnostic_state.add_step("system_error", {"error": error_message})
            diagnostic_state.final_diagnosis = f"Triage failed due to system error: {e}"

        return diagnostic_state


# --- FastAPI Application ---

app = FastAPI(
    title="LLM Automated Software Issue Triage System",
    description="API for submitting software issues and getting automated triage.",
    version="1.0.0",
)

# --- Global Components ---
# These are created once when the application starts.
# Consider thread safety if components are stateful across requests.
tool_registry = ToolRegistry()
triage_system = TriageSystem(tool_registry=tool_registry)

# Basic in-memory storage for session states (Not suitable for production!)
# Keys are session IDs (UUID strings), values are the DiagnosticState objects
session_states: Dict[str, DiagnosticState] = {}

# --- API Models ---

class TriageRequest(BaseModel):
    problem: str = Field(..., description="The natural language description of the software issue.")

class TriageResponse(BaseModel):
    session_id: str = Field(description="Unique ID for this triage session.")
    initial_problem: str
    steps_taken: List[Dict[str, Any]]
    final_diagnosis: Optional[str]
    conversation_history: List[Dict[str, Any]]

class ToolInfo(BaseModel):
    name: str
    description: str

class ToolDetail(ToolInfo):
     args_schema: Optional[Dict[str, Any]] = Field(description="Pydantic schema for tool arguments")


# --- API Endpoints ---

@app.post("/triage",
          response_model=TriageResponse,
          summary="Submit a Software Issue for Triage",
          status_code=status.HTTP_200_OK)
async def start_triage_session(request: TriageRequest = Body(...)):
    """
    Accepts a natural language problem statement, initiates the triage process,
    and returns the final diagnostic state including the diagnosis or recommendations.
    """
    session_id = str(uuid.uuid4())
    try:
        # Run the triage process
        final_state_obj = triage_system.run_triage(request.problem, session_id)
        final_state_summary = final_state_obj.get_state_summary()

        # Add session_id to the response explicitly if not part of get_state_summary
        response_data = {"session_id": session_id, **final_state_summary}

        return TriageResponse(**response_data)

    except Exception as e:
        logging.error(f"[Session: {session_id}] Failed to process triage request: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An internal error occurred during triage: {e}"
        )

@app.get("/tools",
         response_model=List[ToolInfo],
         summary="List Available Diagnostic Tools")
async def list_tools():
    """Returns a list of names and descriptions of registered diagnostic tools."""
    tools = tool_registry.get_all_tools()
    return [ToolInfo(name=t.name, description=t.description) for t in tools]

@app.get("/tools/{tool_name}",
         response_model=ToolDetail,
         summary="Get Details of a Specific Tool",
         responses={404: {"description": "Tool not found"}})
async def get_tool_details(tool_name: str = Path(..., description="The name of the tool")):
    """Retrieves the description and argument schema for a specific tool."""
    tool = tool_registry.get_tool(tool_name)
    if not tool:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Tool '{tool_name}' not found.")

    schema = None
    if hasattr(tool, 'args_schema') and isinstance(tool.args_schema, type(BaseModel)):
        # Get Pydantic schema definition
        schema = tool.args_schema.schema()

    return ToolDetail(name=tool.name, description=tool.description, args_schema=schema)

@app.get("/state/{session_id}",
         response_model=TriageResponse, # Reusing the same response model
         summary="Retrieve State of a Triage Session",
         responses={404: {"description": "Session not found"}})
async def get_session_state(session_id: str = Path(..., description="The unique ID of the triage session")):
    """
    Retrieves the stored state summary for a completed triage session.
    Requires session state persistence to be effective beyond server restarts.
    """
    diagnostic_state = session_states.get(session_id)
    if not diagnostic_state:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Triage session not found.")

    state_summary = diagnostic_state.get_state_summary()

    # Ensure the response includes the session_id
    response_data = {"session_id": session_id, **state_summary}
    return TriageResponse(**response_data)


# --- Run the Application ---
if __name__ == "__main__":
    # Example: Run using Uvicorn
    # Use `reload=True` for development to automatically reload on code changes
    # Command line: uvicorn app:app --reload --host 0.0.0.0 --port 8000
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)