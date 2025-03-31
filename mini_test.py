# mini_test.py - Manual Tool Creation Test
import sys
import os

print("--- sys.path ---")
for p in sys.path:
    print(p)
print("--- END sys.path ---")

pydantic_v1_paths = [p for p in sys.path if 'pydantic' in p.lower() and os.path.exists(os.path.join(p, 'pydantic', 'v1'))]
print(f"\nPotential Pydantic V1 paths found in sys.path: {pydantic_v1_paths}\n")

import pydantic
from pydantic import BaseModel, Field
# VVVV IMPORTANT: Import the class, not the decorator VVVV
from langchain_core.tools import StructuredTool

print("Python Version:", sys.version)
print("Pydantic Version:", pydantic.__version__)

# Define a simple Pydantic V2 model
class SimpleInput(BaseModel):
    query: str = Field(description="A simple query")

print(f"Is SimpleInput a Pydantic V2 model? {hasattr(SimpleInput, 'model_rebuild')}")

# Define the function
def simple_test_tool_func(query: str) -> str:
     """A very simple test tool."""
     return f"Received query: {query}"

print("Attempting manual StructuredTool creation...")

try:
    # VVVV IMPORTANT: Manually create the StructuredTool instance VVVV
    manual_tool = StructuredTool(
        name="simple_test_tool_manual",
        func=simple_test_tool_func,
        description="A very simple test tool.", # Get from docstring or provide here
        args_schema=SimpleInput
    )

    print("Manual StructuredTool created successfully!")
    print(f"Tool Name: {manual_tool.name}")
    print(f"Tool Arg Schema Set: {manual_tool.args_schema}")

except Exception as e:
    print("\n--- ERROR ---")
    print(f"Failed to create manual tool: {e}")
    import traceback
    traceback.print_exc()
    print("--- END ERROR ---\n")