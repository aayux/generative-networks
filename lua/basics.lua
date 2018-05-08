--[[Lua Torch introductory example
I'll keep adding to it till 
I have the basic blocks covered]]

-- from: https://github.com/torch/nn/blob/master/doc/training.md
-- https://github.com/torch/nn/blob/master/doc/simple.md

require "nn"

--[[StochasticGradient expect as a dataset an object which implements 
  the operator dataset[index] and implements the method dataset:size().
  The size() methods returns the number of examples and dataset[i] 
  has to return the i-th example.]]

dataset = {}
function dataset:size() 
  return 1000 -- 1000 examples
end

for i = 1, dataset:size() do 
  local input = torch.randn(2)     -- normally distributed example in 2d
  local output = torch.Tensor(1)
  if input[1] * input[2] > 0 then     -- calculate label for XOR function
    output[1] = -1
  else
    output[1] = 1
  end
  dataset[i] = {input, output}
end

-- we create a simple neural network with one hidden layer.
mlp = nn.Sequential()  -- make a multi-layer perceptron
inputs = 2; outputs = 1; HUs = 20; -- parameters
mlp:add(nn.Linear(inputs, HUs))
mlp:add(nn.Tanh())
mlp:add(nn.Linear(HUs, outputs))
mlp:add(nn.Tanh())

-- we choose the Mean Squared Error criterion
criterion = nn.MSECriterion()

-- train using SGD
trainer = nn.StochasticGradient(mlp, criterion)
trainer.learningRate = 0.01
trainer:train(dataset)


--[[ For Manual training of parameters

-- train over this example in 3 steps
-- (1) zero the accumulation of the gradients
mlp:zeroGradParameters()
-- (2) accumulate gradients
mlp:backward(input, criterion:backward(mlp.output, output))
-- (3) update parameters with a 0.01 learning rate
mlp:updateParameters(0.01)

-- feed it to the neural network and the criterion
criterion:forward(mlp:forward(input), output)

]]

-- test the network

x = torch.Tensor(2)
x[1] =  0.5; x[2] =  0.5; print(mlp:forward(x))
x[1] =  0.5; x[2] = -0.5; print(mlp:forward(x))
x[1] = -0.5; x[2] =  0.5; print(mlp:forward(x))
x[1] = -0.5; x[2] = -0.5; print(mlp:forward(x))