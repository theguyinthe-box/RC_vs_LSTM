using ReservoirComputing, OrdinaryDiffEq, Plots, Statistics

# Lorenz-System
u0 = [1.0, 0.0, 0.0]
tspan = (0.0, 200.0)
p = [10.0, 28.0, 8 / 3]

function lorenz(du, u, p, t)
    du[1] = p[1] * (u[2] - u[1])
    du[2] = u[1] * (p[2] - u[3]) - u[2]
    du[3] = u[1] * u[2] - p[3] * u[3]
end

prob = ODEProblem(lorenz, u0, tspan, p)
data = Array(solve(prob, ABM54(); dt=0.02))

# Daten vorbereiten
shift = 300
train_len = 5000
predict_len = 1250

input_data = data[:, shift:(shift + train_len - 1)]
target_data = data[:, (shift + 1):(shift + train_len)]
test_data = data[:, (shift + train_len):(shift + train_len + predict_len - 1)]

# Standardisierung (mean=0, std=1)
μ = mean(input_data, dims=2)
σ = std(input_data, dims=2)

function standardize(x)
    (x .- μ) ./ σ
end

function destandardize(x)
    (x .* σ) .+ μ
end

X = standardize(input_data)'  # (n_samples, input_dim)
Y = standardize(target_data)' # (n_samples, output_dim)
test = test_data'             # (predict_len, 3) — so passt es

# Trainings- und Vorhersagefunktion
function train_and_predict(params, X_train, Y_train, predict_len, test_true)
    input_size = size(X_train, 2)
    res_size = params[:res_size]
    sr = params[:sr]
    lr = params[:lr]

    # Reservoir erstellen
    esn = ESN(X_train', input_size, res_size;
        reservoir=rand_sparse(radius=sr, sparsity=6 / res_size),
        input_layer=randn,
        nla_type=NLAT1())

    output_layer = train(esn, Y_train')

    # Generative Vorhersage
    output_scaled = esn(Generative(predict_len), output_layer)
    output_scaled = output_scaled'  # (predict_len, output_dim)

    # destandardisieren
    output = destandardize(output_scaled)

    # Debug-Dimensionen prüfen
    println("output size: ", size(output))        # z.B. (1250, 3)
    println("test_true size: ", size(test_true))  # z.B. (1250, 3)

    # MSE berechnen (über alle Dimensionen und Samples)
    mse = mean((output - test_true).^2)

    return mse, output
end

# Grid Search Parameterbereiche
sr_vals = [1.0, 1.1, 1.2, 1.3]
lr_vals = [0.3, 0.5, 0.7]
units_vals = [300, 400, 500]

# Grid Search in Funktion gekapselt
function run_gridsearch()
    best_loss = Inf
    best_params = nothing
    best_output = nothing
    model_index = 0

    for sr in sr_vals, lr in lr_vals, res_size in units_vals
        params = Dict(:sr => sr, :lr => lr, :res_size => res_size)
        println("[$model_index] Testing: sr=$sr, lr=$lr, units=$res_size")

        try
            loss, output = train_and_predict(params, X, Y, predict_len, test)
            println("   --> MSE: $loss")

            if loss < best_loss
                println("   ✅ New best model found!")
                best_loss = loss
                best_params = params
                best_output = output
            end
        catch e
            println("   ❌ Error during training: $e")
        end

        model_index += 1
    end

    return best_loss, best_params, best_output
end

# Grid Search ausführen
best_loss, best_params, best_output = run_gridsearch()

# Visualisierung der besten Vorhersage
x_pred, y_pred, z_pred = eachcol(best_output)
x_true, y_true, z_true = eachcol(test)

p1 = plot(x_pred, label="predicted x", title="x(t)", xlabel="t", ylabel="x")
plot!(p1, x_true, label="actual x", linestyle=:dash)

p2 = plot(y_pred, label="predicted y", title="y(t)", xlabel="t", ylabel="y")
plot!(p2, y_true, label="actual y", linestyle=:dash)

p3 = plot(z_pred, label="predicted z", title="z(t)", xlabel="t", ylabel="z")
plot!(p3, z_true, label="actual z", linestyle=:dash)

p4 = plot3d(x_pred, y_pred, z_pred, label="predicted", title="3D Trajectory")
plot3d!(p4, x_true, y_true, z_true, label="actual", linestyle=:dash)

plot(p1, p2, p3, p4; layout=(2, 2), size=(900,700))