using ReservoirComputing, OrdinaryDiffEq, Plots, Statistics

# Define Lorenz system
function lorenz(du, u, p, t)
    du[1] = p[1] * (u[2] - u[1])
    du[2] = u[1] * (p[2] - u[3]) - u[2]
    du[3] = u[1] * u[2] - p[3] * u[3]
end

u0 = [1.0, 0.0, 0.0]
tspan = (0.0, 200.0)
p = [10.0, 28.0, 8/3]

prob = ODEProblem(lorenz, u0, tspan, p)
data = Array(solve(prob, ABM54(); dt=0.02))

# Training and test data
shift = 300
train_len = 5000
predict_len = 1250

input_data = data[:, shift:(shift + train_len - 1)]
target_data = data[:, (shift + 1):(shift + train_len)]
test = data[:, (shift + train_len):(shift + train_len + predict_len - 1)]

# Standardize data
μ = mean(input_data, dims=2)
σ = std(input_data, dims=2)
standardize(x) = (x .- μ) ./ σ
destandardize(x) = (x .* σ) .+ μ

X = standardize(input_data)
Y = standardize(target_data)
test_std = standardize(test)

# Initialize ESN
input_size = size(X, 1)
res_size = 300
esn = ESN(X, input_size, res_size;
          reservoir=rand_sparse(; radius=0.9, sparsity=6/res_size),
          input_layer=weighted_init,
          nla_type=NLAT1())

# Train ESN
output_layer = train(esn, Y)

# Prediction (generative)
output_std = esn(Generative(predict_len), output_layer)
output = destandardize(output_std)

# Extract plot data
x_pred = output[1, :]
y_pred = output[2, :]
z_pred = output[3, :]

x_true = test[1, :]
y_true = test[2, :]
z_true = test[3, :]

# Calculate MSE per dimension
using Statistics
mse_x = mean((x_pred - x_true).^2)
mse_y = mean((y_pred - y_true).^2)
mse_z = mean((z_pred - z_true).^2)

# Create plots
p1 = plot(x_pred, label="predicted x", title="x(t), MSE=$(round(mse_x, digits=4))", xlabel="t", ylabel="x")
plot!(p1, x_true, label="actual x", linestyle=:dash)

p2 = plot(y_pred, label="predicted y", title="y(t), MSE=$(round(mse_y, digits=4))", xlabel="t", ylabel="y")
plot!(p2, y_true, label="actual y", linestyle=:dash)

p3 = plot(z_pred, label="predicted z", title="z(t), MSE=$(round(mse_z, digits=4))", xlabel="t", ylabel="z")
plot!(p3, z_true, label="actual z", linestyle=:dash)

p4 = plot3d(x_pred, y_pred, z_pred, label="predicted", title="3D Trajectory")
plot3d!(p4, x_true, y_true, z_true, label="actual", linestyle=:dash)

plot(p1, p2, p3, p4; layout=(2, 2), size=(900,700))
