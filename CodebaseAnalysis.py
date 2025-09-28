# CodebaseAnalysis.py
from pydantic import BaseModel, Field
from typing import List

# Define the structure for a single method
class MethodInfo(BaseModel):
    name: str = Field(description="The exact name of the method.")
    signature: str = Field(description="The full method signature (e.g., public void doSmtg(String s, int i)).")
    description: str = Field(description="A concise description of the method's purpose.")

# Define the overall structure for the codebase analysis
class CodebaseAnalysis(BaseModel):
    project_name: str = Field(description="The name of the analyzed GitHub project.")
    high_level_overview: str = Field(description="A 2-3 sentence summary of the project's purpose and functionality.")
    key_functionality_areas: List[str] = Field(description="A list of 3-4 main functional areas (e.g., 'Database Connection', 'CRUD Operations for Actors').")
    key_methods_summary: List[MethodInfo] = Field(description="A list of the 3-4 most important methods from the codebase.")
    complexity_and_notes: str = Field(description="A brief description of code complexity, notable design patterns, or any limitations.")

# This schema is imported by llm_execution_analysis.py