def change_case(str, delimiter="_") -> str:
    return "".join([delimiter + i.lower() if i.isupper() else i for i in str]).lstrip("_")