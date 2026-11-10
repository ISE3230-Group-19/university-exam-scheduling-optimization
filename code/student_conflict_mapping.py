# Author: Haechan Jung

import numpy as np

# Assume student data
#      = (id, [list of courses enrolled])


# read data from generated course/time
def generate_student_data(df):

    return std_df




def student_conflict_matrix(class_df, std_df):

    # student1 = list of courses enrolled
    def is_conflict(student1, student2):
        for course in student1:
            if course in student2:
                return True
        return False

    cnf_matrix = np.zeros((len(class_df), len(class_df)))
    for i in range(len(cnf_matrix)):
        for j in range(len(cnf_matrix[i])):
            conflict = is_conflict(std_df.iloc[i]['courses'], std_df.iloc[j]['courses'])
            if conflict and i != j:
                cnf_matrix[i, j] = 1

    return cnf_matrix




