import os
import sys
import time
from pathlib import Path
import pandas as pd
from transformers import logging
import transformers.modeling_utils as mutils
from transformers.utils import import_utils


from src.collection    import collect_and_structured
from src.encoding      import build_index
from src.llm_coding    import solve_competition_keras,solve_competition_tuner,followup_prompt
from src.comps         import test

mutils.check_torch_load_is_safe = lambda *args, **kwargs: None
import_utils.check_torch_load_is_safe = lambda *args, **kwargs: None
logging.set_verbosity_error()



# Command‐Line Interface

def print_usage():
    print("""
        Usage:
        python rag.py cb            - Collect notebooks and metadata and build a matrix 
        python rag.py cb <slug>     - Collect notebooks and metadata and build a matrix starting from a certain competition
          
        python rag.py code                 - Build solutions for all the testing competitions in the comps.py and provides one solutions example
        python rag.py code <top-k>         - Build solutions for all the testing competitions in the comps.py and provides k solutions
        python rag.py code <slug>          - Same, just starts from a certain competition, goes down the list and provides one solutions example
        python rag.py code <slug> <top-k>  - Starts from a certain competition and also provides k solutions
    """)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)

    cmd = sys.argv[1].lower()

    if cmd == "cb":
        start_slug = sys.argv[2] if len(sys.argv) >= 3 else None
        df_struct = collect_and_structured(max_per_keyword=5, start=start_slug)
        print(f"[OK] Collected and structured {len(df_struct)} notebooks.")
        if not Path("notebooks_structured.csv").exists():
            print("[ERROR] Please run `collect_and_structured` first.")
            sys.exit(1)
        df_struct = pd.read_csv("notebooks_structured.csv")
        build_index(df_struct)
        sys.exit(0)

    elif cmd == "b":
        if not Path("notebooks_structured.csv").exists():
            print("[ERROR] Please run `collect_and_structured` first.")
            sys.exit(1)
        df_struct = pd.read_csv("notebooks_structured.csv")
        build_index(df_struct)
        sys.exit(0)

    elif cmd == "code":
        if len(sys.argv) < 3:
            print("Usage: python rag.py code <keras-tuner 0|1> [slug - optional] [top-k - optional, default = 1, 1-9]")
            sys.exit(1)

        kt = bool(int(sys.argv[2]))
        comp = None
        top_k = 1
        extra = sys.argv[3:]


        if extra:
            if len(extra) == 1 and extra[0].isdigit():
                top_k = int(extra[0])
            else:
                comp = extra[0]
                if len(extra) > 1 and extra[1].isdigit():
                    top_k = int(extra[1])

        if kt:
            solve = "keras"
            solve = lambda slug: solve_competition_tuner(slug=slug)
            suffix = "kt_solution.py"
        else:
            solve = lambda slug: solve_competition_keras(
                slug=slug,
                structured_csv="notebooks_structured.csv",
                top_k=top_k
            )
            suffix = "solution.py"

        first = comp is None
        for slug in test:
            
            if not first and slug != comp:
                continue
            first = True

            start = time.time()
            print(slug)
            code_str = solve(slug)

            out_path = Path(f"test/{slug}/{slug}_{suffix}")
            out_path.write_text(code_str, encoding="utf-8")

            print(f"[OK] Written to {out_path}")
            print("-" * 27)
            print(time.time() - start)
            print("-" * 27)


    elif cmd == "followup":

        start_time = time.time()

        # Usage: python rag.py followup <slug> <keras-tuner>
        if len(sys.argv) != 4:
            print("Usage: python rag.py followup <keras-tuner 0|1> <slug>")
            sys.exit(1)

        kt         = bool(int(sys.argv[2]))
        slug       = sys.argv[3]
        

        print(slug)
        try:
            corrected_code = followup_prompt(str(slug), kt)
        except Exception as e:
            print(f"[ERROR] {e}")
            sys.exit(1)

        orig = Path("test/"+str(slug)+"/"+str(slug))
        fixed = orig.with_name(orig.stem + "_fixed.py")
        fixed.write_text(corrected_code, encoding="utf-8")
        print(f"[OK] Corrected code written to {fixed}")

        print("---------------------------")
        print(time.time() - start_time)
        print("---------------------------")

    else:
        print_usage()
        sys.exit(1)

