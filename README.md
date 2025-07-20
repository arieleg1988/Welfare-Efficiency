#הקוד בודק את הדאטה סט ומבצע חישוב יעילות לפי VRS ו-CRS לכל שורה ומדפיס את הפלט לקובץ חיצוני#
from pyomo.environ import *

inputs_cols = ['Welfare_Budget', 'Estimated_Social_Workers']  # דוגמה לתשומות
outputs_cols = ['Estimated_Welfare_Clients']                 # דוגמה לתפוקות

# המרה למטריצות numpy (מספרים בלבד)
inputs = df[inputs_cols].astype(float).values
outputs = df[outputs_cols].astype(float).values
n = len(df)

def dea_model(inputs, outputs, i, model_type='VRS'):
    model = ConcreteModel()

    model.lmbda = Var(range(n), domain=NonNegativeReals)
    model.theta = Var(domain=NonNegativeReals)

    model.obj = Objective(expr=model.theta, sense=minimize)

    model.input_constraints = ConstraintList()
    for m in range(inputs.shape[1]):
        model.input_constraints.add(
            sum(model.lmbda[j] * inputs[j, m] for j in range(n)) <= model.theta * inputs[i, m]
        )

    model.output_constraints = ConstraintList()
    for r in range(outputs.shape[1]):
        model.output_constraints.add(
            sum(model.lmbda[j] * outputs[j, r] for j in range(n)) >= outputs[i, r]
        )

    if model_type == 'VRS':
        model.convexity = Constraint(expr=sum(model.lmbda[j] for j in range(n)) == 1)
    elif model_type == 'CRS':
        pass
    else:
        raise ValueError("Invalid model_type. Use 'VRS' or 'CRS'.")

    solver = SolverFactory('glpk')
    result = solver.solve(model, tee=False)

    return model.theta.value

efficiency_vrs = []
efficiency_crs = []

for i in range(n):
    efficiency_vrs.append(dea_model(inputs, outputs, i, 'VRS'))
    efficiency_crs.append(dea_model(inputs, outputs, i, 'CRS'))

df['Efficiency_VRS'] = efficiency_vrs
df['Efficiency_CRS'] = efficiency_crs

print(df[['Efficiency_VRS', 'Efficiency_CRS']])
