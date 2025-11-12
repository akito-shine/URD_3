def convert_pl_to_def(pl_file, def_file, design_name="example_design"):
    components = []
    with open(pl_file, "r") as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.split()
            name = parts[0]
            x = round(float(parts[1]))
            y = round(float(parts[2]))
            orient = parts[-1]
            components.append((name, x, y, orient))

    with open(def_file, "w") as f:
        f.write("VERSION 5.8 ;\n")
        f.write("DIVIDERCHAR \"/\" ;\n")
        f.write("BUSBITCHARS \"[]\" ;\n\n")
        f.write(f"DESIGN {design_name} ;\n\n")
        f.write("UNITS DISTANCE MICRONS 1000 ;\n\n")
        # you can compute DIEAREA based on max X/Y
        max_x = max(c[1] for c in components) + 1000
        max_y = max(c[2] for c in components) + 1000
        f.write(f"DIEAREA ( 0 0 ) ( {max_x} {max_y} ) ;\n\n")

        f.write(f"COMPONENTS {len(components)} ;\n")
        for name, x, y, orient in components:
            f.write(f"  - {name} macro_block + PLACED ( {x} {y} ) {orient} ;\n")
        f.write("END COMPONENTS\n\n")
        f.write("END DESIGN\n")

# Example usage:
convert_pl_to_def("/home/Student113/student_exchange/Shine/EAPlace/pl/adaptec1.pl", "/home/Student113/student_exchange/Shine/EAPlace/pl/placement.def")
