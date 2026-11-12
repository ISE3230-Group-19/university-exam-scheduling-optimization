# Author: Haechan Jung

import pandas as pd
import numpy as np


# input = course conflict data, student enrollment data
# output = conflict matrix: 1 if two courses have shared students, 0 otherwise
def student_conflict_matrix(class_df, std_df):
    courses = std_df.iloc[:, 1:].columns
    courses = courses.to_list()

    # sub-function to check if two courses have shared students
    def is_conflict(c1, c2):
        return (class_df[(class_df['Course 1'] == c1) 
                         & (class_df['Course 2'] == c2)]['Shared Students'] > 0).any()

    # create a conflict matrix with 0s
    base = np.zeros((len(courses), len(courses)))
    cnf_matrix = pd.DataFrame(base, index=courses, columns=courses, dtype=int)
    for c1 in courses:
        for c2 in courses:
            conflict = is_conflict(c1, c2)
            if conflict and c1 != c2:
                cnf_matrix.loc[c1, c2] = 1
                cnf_matrix.loc[c2, c1] = 1
    
    return cnf_matrix

