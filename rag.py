import os
import sys
import time
from pathlib import Path
import pandas as pd
from transformers import logging
import transformers.modeling_utils as mutils
from transformers.utils import import_utils


from collection    import collect_and_structured
from encoding      import build_index
from similarity    import find_similar_ids as find_similar_from_description
from llm_coding    import solve_competition_keras,solve_competition_tuner, followup_prompt
from comps         import test

# Monkey-patches
mutils.check_torch_load_is_safe = lambda *args, **kwargs: None
import_utils.check_torch_load_is_safe = lambda *args, **kwargs: None
logging.set_verbosity_error()



# ─────────────────────────────────────────────────────────────────────────────
# Command‐Line Interface
# ─────────────────────────────────────────────────────────────────────────────

def print_usage():
    print("""
        Usage:
        python rag.py cb            - Collect and build 
        python rag.py find_similar
        python rag.py find_similar_desc <description.txt> [notebooks|comps] [top_k]

        Examples:
        python rag.py find_similar_desc my_problem.txt notebooks 5
        python rag.py find_similar_desc my_problem.txt comps 10
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
    elif cmd == 'b':
        if not Path("notebooks_structured.csv").exists():
            print("[ERROR] Please run `collect_and_structured` first.")
            sys.exit(1)
        df_struct = pd.read_csv("notebooks_structured.csv")
        build_index(df_struct)
        sys.exit(0)
    elif cmd == "fd":
        # Usage: python3 rag.py fd <desc_and_meta.json> <top_k> [<exclude_competition>]
        if len(sys.argv) < 4:
            print("Usage: python3 rag.py fd <desc_meta.json> <top_k> [<exclude_competition>]")
            sys.exit(1)

        desc_json    = sys.argv[2]
        top_k        = int(sys.argv[3])
        exclude_comp = sys.argv[4] if len(sys.argv) >= 5 else None

        find_similar_from_description(
            desc_json,
            top_k=top_k,
            exclude_competition=exclude_comp
        )
        sys.exit(0)

    elif cmd == "code":
        if len(sys.argv) < 4:
            print("Usage: python rag.py code <top-k> <keras-tuner 0|1> <slug: optional, to start at a certain competition>")
            sys.exit(1)

        start_time = time.time()

        top_k = int(sys.argv[2])  
        kt    = bool(int(sys.argv[3]))
        comp = None if len(sys.argv) < 5 else sys.argv[4]

        run = 1 if comp is None else 0

        if kt == 0: 
            for slug in test: 
                if(slug == "conway-s-reverse-game-of-life_solution"): break

                if run == 1 or slug == comp: 
                    run = 1
                    print(slug)
                    notebook_code = solve_competition_keras(
                        slug = slug,
                        structured_csv= "notebooks_structured.csv",
                        top_k         = top_k
                    )
                    out_path = Path(f"test/{slug}/{slug}_solution.py")
                    out_path.write_text(notebook_code, encoding="utf-8")
                    print(f"[OK] Solution code written to {out_path}")
                    print("---------------------------")
                    print(start_time - time.time())
                    print("---------------------------")
        else: 
            for slug in test: 
                if run == 1 or slug == comp: 
                    run = 1
                    print(slug)
                    notebook_code = solve_competition_tuner(
                        slug = slug, 
                    )
                    out_path = Path(f"test/{slug}/{slug}_kt_solution.py")
                    out_path.write_text(notebook_code, encoding="utf-8")
                    print(f"[OK] Solution code written to {out_path}")
                    print("---------------------------")
                    print(start_time - time.time())
                    print("---------------------------")





    elif cmd == "followup":

        start_time = time.time()

        # Usage: python rag.py followup <slug> <keras-tuner>
        if len(sys.argv) != 4:
            print("Usage: python rag.py followup <slug> <keras-tuner 0|1>")
            sys.exit(1)

        slug       = sys.argv[2]
        kt         = bool(int(sys.argv[3]))

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
        print(start_time - time.time())
        print("---------------------------")

    else:
        print_usage()
        sys.exit(1)

