import json
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import os


# --- 1. Conceptual Representation Interface (CRI) ---
@dataclass
class Object:
    """Basic object representation in CRI"""

    id: str
    shape: str
    colors: List[int]  # Unique colors present in the grid_section
    position: Tuple[int, int]  # (row_start, col_start) of its bounding box
    size: Tuple[int, int]  # (height, width) of its bounding box
    grid_section: np.ndarray  # The actual numpy array segment of the object
    # For MVP, spatial_relations will be simplified or derived externally
    # For a full MVP, more properties could be added like 'is_symmetric', 'orientation'


@dataclass
class ConceptualRepresentation:
    """CRI - Conceptual Representation Interface"""

    objects: List[Object]
    grid_size: Tuple[int, int]  # Overall grid (height, width)
    color_mapping: Dict[
        int, str
    ]  # Map color integer to descriptive string (e.g., "color_7")
    spatial_relations: Dict[str, Any]  # Placeholder for broader relations


# --- 2. Visual Parser (Perception -> CRI) ---
class VisualParser:
    """Converts raw grids to conceptual representations by detecting objects."""

    def parse_grid(self, grid: List[List[int]]) -> ConceptualRepresentation:
        """Parse a grid into conceptual representation by identifying objects."""
        grid_array = np.array(grid, dtype=int)
        height, width = grid_array.shape

        objects = []
        visited = np.zeros_like(grid_array, dtype=bool)
        object_id_counter = 0
        all_colors = np.unique(grid_array).tolist()

        # Simple BFS/DFS for connected components (objects)
        for r in range(height):
            for c in range(width):
                if (
                    not visited[r, c] and grid_array[r, c] != 0
                ):  # Assuming 0 is background
                    object_id_counter += 1
                    obj_id = f"obj_{object_id_counter}"
                    component_pixels = []
                    queue = [(r, c)]
                    visited[r, c] = True
                    min_r, max_r = r, r
                    min_c, max_c = c, c

                    while queue:
                        curr_r, curr_c = queue.pop(0)
                        component_pixels.append((curr_r, curr_c))
                        min_r = min(min_r, curr_r)
                        max_r = max(max_r, curr_r)
                        min_c = min(min_c, curr_c)
                        max_c = max(max_c, curr_c)

                        for dr, dc in [
                            (0, 1),
                            (0, -1),
                            (1, 0),
                            (-1, 0),
                        ]:  # 4-connectivity
                            nr, nc = curr_r + dr, curr_c + dc
                            if (
                                0 <= nr < height
                                and 0 <= nc < width
                                and not visited[nr, nc]
                                and grid_array[nr, nc] == grid_array[r, c]
                            ):
                                visited[nr, nc] = True
                                queue.append((nr, nc))

                    # Extract object properties
                    obj_height = max_r - min_r + 1
                    obj_width = max_c - min_c + 1
                    obj_grid_section = grid_array[
                        min_r : max_r + 1, min_c : max_c + 1
                    ]
                    obj_colors = np.unique(obj_grid_section).tolist()

                    obj = Object(
                        id=obj_id,
                        shape="rectangular_component",  # Simplified shape for MVP
                        colors=obj_colors,
                        position=(min_r, min_c),
                        size=(obj_height, obj_width),
                        grid_section=obj_grid_section,
                    )
                    objects.append(obj)

        # Fallback for patterns that might be seen as a single large object
        if not objects or (
            len(objects) == 1 and objects[0].size == (height, width)
        ):
            # Re-evaluate as potentially a single large pattern if no smaller objects found or only one covering the whole grid
            obj = Object(
                id="full_grid_pattern",
                shape="full_grid_pattern",
                colors=list(np.unique(grid_array)),
                position=(0, 0),
                size=(height, width),
                grid_section=grid_array,
            )
            objects = [
                obj
            ]  # Overwrite, ensuring at least one object exists representing the whole grid

        return ConceptualRepresentation(
            objects=objects,
            grid_size=(height, width),
            color_mapping={color: f"color_{color}" for color in all_colors},
            spatial_relations={},  # Simplified for MVP
        )


# --- Rule Representation Language (RRL) ---
class RuleRepresentation:
    """Rule Representation Language (RRL) for MVP"""

    def __init__(self, rule_type: str, parameters: Dict[str, Any]):
        self.rule_type = rule_type
        self.parameters = parameters

    def __repr__(self):
        return f"Rule({self.rule_type}, {self.parameters})"

    def __eq__(self, other):
        if not isinstance(other, RuleRepresentation):
            return NotImplemented
        return (
            self.rule_type == other.rule_type
            and self.parameters == other.parameters
        )

    def __hash__(self):
        # Make it hashable for sets/dicts if needed, by converting parameters to a frozenset of items
        return hash((self.rule_type, frozenset(self.parameters.items())))


# --- 3. Rule Executor (RE) ---
class RuleExecutor:
    """Executes symbolic rules on conceptual representations"""

    def execute(
        self, rule: RuleRepresentation, input_cri: ConceptualRepresentation
    ) -> np.ndarray:
        """Execute a rule on input CRI to produce output grid"""

        # Dispatch based on rule type. Add more rule types as your RuleProposer proposes them.
        if (
            rule.rule_type == "replicate_and_alternate_row_flip"
        ):  # New specific rule name
            return self._replicate_and_alternate_row_flip(
                input_cri, rule.parameters
            )
        elif rule.rule_type == "simple_replicate":
            return self._simple_replicate(input_cri, rule.parameters)
        elif rule.rule_type == "color_swap":
            return self._color_swap(input_cri, rule.parameters)
        elif rule.rule_type == "add_border":
            return self._add_border(input_cri, rule.parameters)
        else:
            raise ValueError(f"Unknown rule type: {rule.rule_type}")

    def _replicate_and_alternate_row_flip(
        self, cri: ConceptualRepresentation, params: Dict
    ) -> np.ndarray:
        """Replicate pattern horizontally and vertically with row-wise horizontal flip"""
        # Assumes input_cri has one main pattern object (e.g., from full_grid_pattern)
        if not cri.objects or cri.objects[0].shape not in [
            "rectangular_pattern",
            "full_grid_pattern",
        ]:
            raise ValueError(
                "Input CRI must contain a primary pattern object for replication."
            )

        base_pattern = cri.objects[0].grid_section
        h_reps = params["horizontal_reps"]
        v_reps = params["vertical_reps"]

        base_h, base_w = base_pattern.shape

        output_grid = np.zeros((base_h * v_reps, base_w * h_reps), dtype=int)

        for v in range(v_reps):
            for h in range(h_reps):
                pattern_instance = base_pattern

                # Apply horizontal flip if the row index 'v' is odd
                if v % 2 == 1:
                    pattern_instance = np.flip(
                        base_pattern, axis=1
                    )  # Only flip horizontally

                start_row = v * base_h
                start_col = h * base_w
                output_grid[
                    start_row : start_row + base_h,
                    start_col : start_col + base_w,
                ] = pattern_instance
        return output_grid

    def _simple_replicate(
        self, cri: ConceptualRepresentation, params: Dict
    ) -> np.ndarray:
        """Simple replication without alternation"""
        if not cri.objects or cri.objects[0].shape not in [
            "rectangular_pattern",
            "full_grid_pattern",
        ]:
            raise ValueError(
                "Input CRI must contain a primary pattern object for replication."
            )

        base_pattern = cri.objects[0].grid_section
        h_reps = params["horizontal_reps"]
        v_reps = params["vertical_reps"]
        return np.tile(base_pattern, (v_reps, h_reps))

    def _color_swap(
        self, cri: ConceptualRepresentation, params: Dict
    ) -> np.ndarray:
        """Swaps all occurrences of one color with another."""
        input_grid = np.zeros(cri.grid_size, dtype=int)
        # Reconstruct input grid from CRI objects (simplified, assuming full grid initially)
        if (
            cri.objects
        ):  # Assuming the first object represents the full grid if only one
            input_grid = cri.objects[0].grid_section.copy()

        old_color = params["old_color"]
        new_color = params["new_color"]

        output_grid = input_grid.copy()
        output_grid[output_grid == old_color] = new_color
        return output_grid

    def _add_border(
        self, cri: ConceptualRepresentation, params: Dict
    ) -> np.ndarray:
        """Adds a border of a specified color and thickness to the grid."""
        input_grid = np.zeros(cri.grid_size, dtype=int)
        # Reconstruct input grid from CRI objects (simplified, assuming full grid initially)
        if (
            cri.objects
        ):  # Assuming the first object represents the full grid if only one
            input_grid = cri.objects[0].grid_section.copy()

        border_color = params["color"]
        thickness = params.get("thickness", 1)  # Default thickness 1

        new_height = cri.grid_size[0] + 2 * thickness
        new_width = cri.grid_size[1] + 2 * thickness
        output_grid = np.full((new_height, new_width), border_color, dtype=int)

        # Place the original grid in the center
        output_grid[thickness:-thickness, thickness:-thickness] = input_grid
        return output_grid


# --- 4. Rule Proposer (RP) ---
class RuleProposer:
    """Proposes transformation rules based on input/output examples (heuristic-based MVP)."""

    def propose_rules(
        self,
        input_cri: ConceptualRepresentation,
        output_cri: ConceptualRepresentation,
    ) -> List[RuleRepresentation]:
        """Propose rules that could transform input to output."""
        rules = []

        # Heuristic 1: Check for Replication Patterns
        if input_cri.objects and output_cri.objects:
            input_obj = input_cri.objects[
                0
            ]  # Assuming first object is the main pattern
            output_obj = output_cri.objects[0]

            input_shape = input_obj.size
            output_shape = output_obj.size

            if (
                output_shape[0] % input_shape[0] == 0
                and output_shape[1] % input_shape[1] == 0
            ):
                v_reps = output_shape[0] // input_shape[0]
                h_reps = output_shape[1] // input_shape[1]

                # Proposed Rule 1.1: Simple Replication
                rules.append(
                    RuleRepresentation(
                        "simple_replicate",
                        {"horizontal_reps": h_reps, "vertical_reps": v_reps},
                    )
                )

                # Proposed Rule 1.2: Replication with Row-wise Horizontal Flip
                # Check if it *might* be this pattern based on dimensions and color changes
                # (A more advanced check would compare actual content, but for proposer, just propose)
                if (
                    v_reps > 1
                ):  # Only makes sense if there's more than one row of patterns
                    rules.append(
                        RuleRepresentation(
                            "replicate_and_alternate_row_flip",
                            {
                                "horizontal_reps": h_reps,
                                "vertical_reps": v_reps,
                            },
                        )
                    )

        # Heuristic 2: Check for Color Swaps (assuming only one object, the full grid)
        if len(input_cri.objects) == 1 and len(output_cri.objects) == 1:
            input_colors = set(input_cri.objects[0].colors)
            output_colors = set(output_cri.objects[0].colors)

            # Simple check: if two sets of colors are the same size, and elements swapped
            if len(input_colors) == len(output_colors):
                for old_c in input_colors:
                    for new_c in output_colors:
                        if old_c != new_c:
                            # Propose a swap for every possible pair (will be validated later)
                            rules.append(
                                RuleRepresentation(
                                    "color_swap",
                                    {"old_color": old_c, "new_color": new_c},
                                )
                            )

        # Heuristic 3: Check for Border Addition
        # If output is larger than input by 2 in both dimensions, and outer layer is uniform color
        if (
            output_cri.grid_size[0] == input_cri.grid_size[0] + 2
            and output_cri.grid_size[1] == input_cri.grid_size[1] + 2
        ):

            # This is a very simplified check assuming full grid is parsed as one object
            # and that the border color is the only new color or dominates the edge.
            # A more robust check would involve checking actual border pixel values.

            # Propose a border if a new color appears only at the edges
            new_colors = set(output_cri.color_mapping.keys()) - set(
                input_cri.color_mapping.keys()
            )
            if len(new_colors) == 1:
                rules.append(
                    RuleRepresentation(
                        "add_border",
                        {"color": list(new_colors)[0], "thickness": 1},
                    )
                )

        return rules


# --- 5. Rule Refiner (RR) ---
class RuleRefiner:
    """Refines and validates proposed rules against all training examples."""

    def __init__(
        self, rule_executor: RuleExecutor, visual_parser: VisualParser
    ):
        self.rule_executor = rule_executor
        self.visual_parser = visual_parser

    def refine_and_validate(
        self,
        candidate_rules: List[RuleRepresentation],
        training_examples: List[Dict[str, Any]],
    ) -> List[RuleRepresentation]:
        """
        Validates candidate rules against all training examples.
        For MVP, refinement means discarding rules that don't work perfectly.
        """
        valid_rules = []

        for rule in candidate_rules:
            is_valid = True
            for example in training_examples:
                input_cri_val = self.visual_parser.parse_grid(example["input"])
                expected_output = np.array(example["output"], dtype=int)

                try:
                    predicted_output = self.rule_executor.execute(
                        rule, input_cri_val
                    )
                    # Check if shapes match before comparing content
                    if (
                        predicted_output.shape != expected_output.shape
                        or not np.array_equal(
                            predicted_output, expected_output
                        )
                    ):
                        is_valid = False
                        break
                except (
                    ValueError
                ) as e:  # Catch errors from executor if rule is not applicable
                    is_valid = False
                    # print(f"Rule {rule.rule_type} failed execution for example: {e}") # For debugging
                    break
                except Exception as e:  # Catch other unexpected errors
                    # print(f"Unexpected error executing rule {rule}: {e}") # For debugging
                    is_valid = False
                    break

            if is_valid:
                valid_rules.append(rule)
        return valid_rules


# --- 6. Memory Control System (MCS) ---
class MemoryControlSystem:
    """Simple cache for successfully learned rules (MVP)."""

    def __init__(self):
        self.learned_rules_cache: List[RuleRepresentation] = []

    def store_rule(self, rule: RuleRepresentation):
        """Stores a successfully learned rule."""
        if rule not in self.learned_rules_cache:
            self.learned_rules_cache.append(rule)
            # print(f"Stored rule in memory: {rule}") # For debugging

    def retrieve_relevant_rules(
        self, input_cri: ConceptualRepresentation
    ) -> List[RuleRepresentation]:
        """
        Retrieves rules that might be relevant based on input characteristics.
        (Very simplified for MVP).
        """
        # For MVP, just return all cached rules.
        # A more sophisticated MCS would use input_cri to filter.
        return list(self.learned_rules_cache)


class ARCDataLoader:
    """Loads and manages ARC task data"""

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)

    def load_task(self, task_id: str) -> Dict:
        """Load a specific task"""
        task_path = self.data_dir / f"{task_id}.json"
        with open(task_path, "r") as f:
            return json.load(f)

    def get_all_task_ids(self) -> List[str]:
        """Get all available task IDs"""
        task_files = list(self.data_dir.glob("*.json"))
        return [f.stem for f in task_files]


# --- Main MIXAL System ---
class MIXALCore:
    """Main MIXAL system - MVP version with all simplified components."""

    def __init__(self):
        self.parser = VisualParser()
        self.rule_executor = RuleExecutor()
        self.rule_proposer = RuleProposer()
        self.rule_refiner = RuleRefiner(self.rule_executor, self.parser)
        self.memory_control_system = MemoryControlSystem()

    def solve_task(self, task_data: Dict) -> Dict:
        """Solve an ARC task by going through the MIXAL pipeline."""
        training_examples = task_data["train"]
        test_input_grid = task_data["test"][0]["input"]

        # Step 1: Parse first training example to get initial hypothesis space
        # In a more advanced system, all examples would inform proposal,
        # but for MVP, we simplify.
        first_input_cri = self.parser.parse_grid(training_examples[0]["input"])
        first_output_cri = self.parser.parse_grid(
            training_examples[0]["output"]
        )

        # Step 2: Propose rules based on first example
        candidate_rules = self.rule_proposer.propose_rules(
            first_input_cri, first_output_cri
        )

        # Optionally, retrieve rules from memory and add to candidates (simplistic for MVP)
        # For more complex MCS, retrieval would be smarter.
        # candidate_rules.extend(self.memory_control_system.retrieve_relevant_rules(first_input_cri))
        # candidate_rules = list(set(candidate_rules)) # Remove duplicates

        # Step 3: Refine and validate rules on all training examples
        valid_rules = self.rule_refiner.refine_and_validate(
            candidate_rules, training_examples
        )

        # Step 4: Apply best rule to test input and store in memory
        if valid_rules:
            # For MVP, simply take the first valid rule
            best_rule = valid_rules[0]
            self.memory_control_system.store_rule(
                best_rule
            )  # Store the learned rule

            test_input_cri = self.parser.parse_grid(test_input_grid)
            predicted_output = self.rule_executor.execute(
                best_rule, test_input_cri
            )

            return {
                "success": True,
                "rule": best_rule,
                "prediction": predicted_output.tolist(),
                "explanation": f"Applied rule: {best_rule}",
            }
        else:
            return {
                "success": False,
                "rule": None,
                "prediction": None,
                "explanation": "No valid rules found for this task.",
            }


# --- Example Usage (from previous iteration) ---
def main():
    """Test the MVP on the provided example"""

    # Test data from your example
    test_task_replication = {
        "train": [
            {
                "input": [[7, 9], [4, 3]],
                "output": [
                    [7, 9, 7, 9, 7, 9],
                    [4, 3, 4, 3, 4, 3],
                    [9, 7, 9, 7, 9, 7],
                    [3, 4, 3, 4, 3, 4],
                    [7, 9, 7, 9, 7, 9],
                    [4, 3, 4, 3, 4, 3],
                ],
            },
            {
                "input": [[8, 6], [6, 4]],
                "output": [
                    [8, 6, 8, 6, 8, 6],
                    [6, 4, 6, 4, 6, 4],
                    [6, 8, 6, 8, 6, 8],
                    [4, 6, 4, 6, 4, 6],
                    [8, 6, 8, 6, 8, 6],
                    [6, 4, 6, 4, 6, 4],
                ],
            },
        ],
        "test": [
            {
                "input": [[3, 2], [7, 8]],
                "output": [
                    [3, 2, 3, 2, 3, 2],
                    [7, 8, 7, 8, 7, 8],
                    [2, 3, 2, 3, 2, 3],
                    [8, 7, 8, 7, 8, 7],
                    [3, 2, 3, 2, 3, 2],
                    [7, 8, 7, 8, 7, 8],
                ],
            }
        ],
    }

    # Example 2: Color Swap (Hypothetical task)
    test_task_color_swap = {
        "train": [
            {
                "input": [[1, 1, 2], [1, 2, 2], [3, 3, 3]],
                "output": [[5, 5, 2], [5, 2, 2], [3, 3, 3]],  # Swap 1s to 5s
            }
        ],
        "test": [
            {
                "input": [[1, 4, 1], [4, 1, 4], [4, 4, 1]],
                "output": [
                    [5, 4, 5],
                    [4, 5, 4],
                    [4, 4, 5],
                ],  # Expect 1s to be 5s
            }
        ],
    }

    # Example 3: Add Border (Hypothetical task)
    test_task_add_border = {
        "train": [
            {
                "input": [[1, 1], [1, 1]],
                "output": [
                    [0, 0, 0, 0],
                    [0, 1, 1, 0],
                    [0, 1, 1, 0],
                    [0, 0, 0, 0],
                ],  # Add 0 border
            }
        ],
        "test": [
            {
                "input": [[2, 2, 2], [2, 2, 2], [2, 2, 2]],
                "output": [
                    [9, 9, 9, 9, 9],
                    [9, 2, 2, 2, 9],
                    [9, 2, 2, 2, 9],
                    [9, 2, 2, 2, 9],
                    [9, 9, 9, 9, 9],
                ],  # Add 9 border
            }
        ],
    }

    print("--- Testing Replication Task ---")
    mixal = MIXALCore()
    result_replication = mixal.solve_task(test_task_replication)
    print(f"Success: {result_replication['success']}")
    print(f"Rule: {result_replication['rule']}")
    print(f"Explanation: {result_replication['explanation']}")
    if result_replication["success"]:
        print("\nPredicted Output:")
        print(np.array(result_replication["prediction"]))
        print("\nExpected Output:")
        print(np.array(test_task_replication["test"][0]["output"]))
        print(
            f"\nCorrect: {np.array_equal(np.array(result_replication['prediction']), np.array(test_task_replication['test'][0]['output']))}"
        )
    print("-" * 30)

    print("\n--- Testing Color Swap Task ---")
    mixal_cs = MIXALCore()  # Re-initialize to clear memory for new task type
    result_color_swap = mixal_cs.solve_task(test_task_color_swap)
    print(f"Success: {result_color_swap['success']}")
    print(f"Rule: {result_color_swap['rule']}")
    print(f"Explanation: {result_color_swap['explanation']}")
    if result_color_swap["success"]:
        print("\nPredicted Output:")
        print(np.array(result_color_swap["prediction"]))
        print("\nExpected Output:")
        print(np.array(test_task_color_swap["test"][0]["output"]))
        print(
            f"\nCorrect: {np.array_equal(np.array(result_color_swap['prediction']), np.array(test_task_color_swap['test'][0]['output']))}"
        )
    print("-" * 30)

    print("\n--- Testing Add Border Task ---")
    mixal_border = MIXALCore()  # Re-initialize
    result_add_border = mixal_border.solve_task(test_task_add_border)
    print(f"Success: {result_add_border['success']}")
    print(f"Rule: {result_add_border['rule']}")
    print(f"Explanation: {result_add_border['explanation']}")
    if result_add_border["success"]:
        print("\nPredicted Output:")
        print(np.array(result_add_border["prediction"]))
        print("\nExpected Output:")
        print(np.array(test_task_add_border["test"][0]["output"]))
        print(
            f"\nCorrect: {np.array_equal(np.array(result_add_border['prediction']), np.array(test_task_add_border['test'][0]['output']))}"
        )
    print("-" * 30)


if __name__ == "__main__":
    main()
