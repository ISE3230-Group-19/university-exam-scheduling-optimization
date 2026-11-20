# modified OptimizationModel_modified.ipynb for testing

import random
import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import os



def opmiziation_modeling_testing(slots, capacity, csv_name):
    # ------------------------------------------------------------
    # Random seeds
    # ------------------------------------------------------------
    np.random.seed(42)
    random.seed(42)
    
    # ------------------------------------------------------------
    # 1. Sets & basic parameters
    # ------------------------------------------------------------
    numberCourses = 25      
    numberRooms = 10        
    numberSlots = slots       
    numberDays = 4
    slotsPerDay = 6
    numberStudents = 400    # Scaled student pool for realistic conflict resolution

    courses = [f"CS_{100 + i*10}" for i in range(numberCourses)]
    rooms = [f"Room_{101 + i}" for i in range(numberRooms)]
    timeSlots = [f"Slot_{t+1}" for t in range(numberSlots)]

    dayOfSlot = {timeSlots[t]: t // slotsPerDay + 1 for t in range(numberSlots)}

    # count adjacent slots only when both slots belong to the same day.
    adjSlots = [
        (timeSlots[t], timeSlots[t + 1])
        for t in range(numberSlots - 1)
        if (t + 1) % slotsPerDay != 0
    ]

    # ------------------------------------------------------------
    # 2. Course enrollments & room capacities
    # ------------------------------------------------------------
    # Generous room capacities to avoid room-size bottlenecks
    # capacityList = [60, 80, 100, 120, 140, 160, 180, 200, 220, 250] 
    capacityList = capacity
    roomCapacities = {rooms[i]: capacityList[i] for i in range(numberRooms)}

    # ------------------------------------------------------------
    # 3. Simulate student and course enrollments
    # ------------------------------------------------------------
    studentID = [f"S_{1000 + i}" for i in range(numberStudents)]
    enrollmentMatrix = np.zeros((numberStudents, numberCourses), dtype=int)

    for s_idx, s in enumerate(studentID):
        enrolled_course_indices = np.random.choice(numberCourses, size=np.random.randint(3, 5), replace=False)
        enrollmentMatrix[s_idx, enrolled_course_indices] = 1

    studentCourseDataFrame = pd.DataFrame(enrollmentMatrix, index=studentID, columns=courses)

    # match the enrollment data and random-number sequence of the modified model.
    courseEnrollments = {
        c: int(studentCourseDataFrame[c].sum()) for c in courses
    }
    studentCourses = {
        s: studentCourseDataFrame.columns[
            studentCourseDataFrame.loc[s] == 1
        ].tolist()
        for s in studentID
    }

    # ------------------------------------------------------------
    # 4. Course to course conflict parameter
    # ------------------------------------------------------------
    overlap_dict = {}
    for i, c1 in enumerate(courses):
        for j, c2 in enumerate(courses):
            if i < j:
                shared_students = np.sum(studentCourseDataFrame[c1] & studentCourseDataFrame[c2])
                if shared_students > 0:
                    overlap_dict[(c1, c2)] = int(shared_students)

    print("--- Data Simulation Complete ---")
    print(f"Courses: {numberCourses} | Rooms: {numberRooms} | Time Slots: {numberSlots} | Students: {numberStudents}")


    # ============================================================
    # 5. OPTIMIZATION MODEL
    # ============================================================

    model = gp.Model("University_Exam_Scheduling")
    model.Params.LogToConsole = 0

    # enforce room capacity by creating variables only for valid assignments.
    valid_crt = [
        (c, r, t)
        for c in courses
        for r in rooms
        if courseEnrollments[c] <= roomCapacities[r]
        for t in timeSlots
    ]
    valid_rooms_for_course = {
        c: [r for r in rooms if courseEnrollments[c] <= roomCapacities[r]]
        for c in courses
    }

    # x assigns exams; y tracks a student's same-day back-to-back exams.
    x = model.addVars(valid_crt, vtype=GRB.BINARY, name="x")
    adj_slot_starts = [t1 for t1, _ in adjSlots]
    y = model.addVars(studentID, adj_slot_starts, vtype=GRB.BINARY, name="y")

    # --- Objective Function ---
    model.setObjective(
        gp.quicksum(y[s, t1] for s in studentID for t1 in adj_slot_starts),
        GRB.MINIMIZE,
    )

    # --- Constraints ---
    # Constraint 1: Each course gets exactly one time slot and room assignment
    model.addConstrs(
        (gp.quicksum(
            x[c, r, t] for r in valid_rooms_for_course[c] for t in timeSlots
        ) == 1 for c in courses),
        name="OneAssignmentPerCourse"
    )

    # Constraint 3: The classroom must be empty during time slot (This means at most 1 exam per room per slot)
    model.addConstrs(
        (gp.quicksum(
            x[c, r, t] for c in courses
            if courseEnrollments[c] <= roomCapacities[r]
        ) <= 1 for r in rooms for t in timeSlots),
        name="RoomMutex"
    )

    # prevent simultaneous exams for course pairs that share students.
    model.addConstrs(
        (gp.quicksum(x[c1, r, t] for r in valid_rooms_for_course[c1])
         + gp.quicksum(x[c2, r, t] for r in valid_rooms_for_course[c2]) <= 1
         for (c1, c2) in overlap_dict for t in timeSlots),
        name="PairwiseConflict",
    )

    # Penalize a student's exams in adjacent slots on the same day.
    model.addConstrs(
        (gp.quicksum(
            x[c, r, t1] + x[c, r, t2]
            for c in studentCourses[s]
            for r in valid_rooms_for_course[c]
        ) - 1 <= y[s, t1]
         for s in studentID for (t1, t2) in adjSlots
         if len(studentCourses[s]) >= 2),
        name="BackToBackTracking",
    )

    # ------------------------------------------------------------
    # 6. Solve Model
    # ------------------------------------------------------------
    print("Optimizing model with Gurobi...")
    # Use the same stopping criteria for every sensitivity scenario.
    model.Params.TimeLimit = 15
    model.Params.MIPGap = 0.05
    model.optimize()

    # ------------------------------------------------------------
    # 7. Export Results & Outputs
    # ------------------------------------------------------------
    os.makedirs("data", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    studentCourseDataFrame.to_csv("data/student_enrollments.csv")
    conflictDataFrame = pd.DataFrame(
        [(c1, c2, count) for (c1, c2), count in overlap_dict.items()],
        columns=["Course 1", "Course 2", "Shared Students"]
    )
    conflictDataFrame.to_csv("data/course_conflicts.csv", index=False)

    df_schedule = None
    test_path = "outputs/" + csv_name
    if model.SolCount > 0:
        print(f"\nSolution found! Status: {model.Status}")
        print(f"Back-to-back objective: {model.ObjVal:.0f}")
        print(f"Relative MIP gap: {model.MIPGap:.2%}")
        
        schedule_records = []
        for c, r, t in valid_crt:
            if x[c, r, t].X > 0.5:
                schedule_records.append({
                    "Course": c,
                    "Enrollment": courseEnrollments[c],
                    "Assigned_Room": r,
                    "Room_Capacity": roomCapacities[r],
                    "Time_Slot": t,
                    "Day": dayOfSlot[t]
                })
                        
        df_schedule = pd.DataFrame(schedule_records)
        df_schedule.to_csv(test_path, index=False)
        print(f"Exam schedule exported to '{test_path}'!")
        print(df_schedule.head(10))
    else:
        print(f"\nOptimization finished with status {model.status}.")
        # remove an old schedule if this scenario no longer has a solution.
        if os.path.isfile(test_path):
            os.remove(test_path)

    result = {
        "slots": slots,
        "capacity": list(capacity),
        "status": model.Status,
        "runtime": model.Runtime,
        "has_solution": model.SolCount > 0,
        "schedule": df_schedule,
        # keep unavailable solution metrics empty instead of using zero.
        "objective": model.ObjVal if model.SolCount > 0 else None,
        "best_bound": model.ObjBound if model.SolCount > 0 else None,
        "mip_gap": model.MIPGap if model.SolCount > 0 else None,
    }

    model.dispose()
    return result
