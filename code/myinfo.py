"""Student information.

They are used for the name of the result file:
    result/result_<student ID>_<name>.csv
"""


def get_info():
    """Return your name and student ID as a dict of two strings."""
    name = "강준현"
    student_id = "20240674"
    return {"name": name, "student_id": student_id}
