import pandas as pd, matplotlib.pyplot as plt

d = pd.read_csv("student.csv")

print("Mean of marks according to subjects:\n", d.groupby("Subjects")["Marks"].mean(), "\n")
print("Median of marks according to subjects:\n", d.groupby("Subjects")["Marks"].median(), "\n")
print("Mode of marks:\n", d["Marks"].mode(), "\n")
print("Variance of marks according to subjects:\n", d.groupby("Subjects")["Marks"].var(), "\n")
print("Standard deviation of marks according to subjects:\n", d.groupby("Subjects")["Marks"].std(), "\n")

c = d["Marks"].corr(d["Attendance"])
print("Correlation between marks & attendance:\n", c)

plt.scatter(d["Attendance"], d["Marks"])
plt.title("Marks vs Attendance")
plt.xlabel("Attendance (out of 100)")
plt.ylabel("Marks")
plt.show()
