
-- Requires


local grad = require 'autograd'
local util = require 'autograd.util'
local lossFuns = require 'autograd.loss'
local optim = require 'optim'



-- for CNNs, we rely on efficient nn-provided primitives:
-- local conv,params,bn,acts,pool = {},{},{},{},{}
-- local flatten,linear

-- Ensure same init on all nodes:
torch.manualSeed(0)

-- for CNNs, we rely on efficient nn-provided primitives:
local reshape = grad.nn.Reshape(1,32,32)

local conv1, acts1, pool1, conv2, acts2, pool2, flatten, linear
local params = {}
conv1, params.conv1 = grad.nn.SpatialConvolutionMM(1, 16, 5, 5)
acts1 = grad.nn.Tanh()
pool1 = grad.nn.SpatialMaxPooling(2, 2, 2, 2)

conv2, params.conv2 = grad.nn.SpatialConvolutionMM(16, 16, 5, 5)
acts2 = grad.nn.Tanh()
pool2, params.pool2 = grad.nn.SpatialMaxPooling(2, 2, 2, 2)

flatten = grad.nn.Reshape(16*5*5)
linear,params.linear = grad.nn.Linear(16*5*5, 10)

-- Cast the parameters
params = grad.util.cast(params, 'float')

-- Make sure all the nodes have the same parameter values
allReduceSGD.synchronizeParameters(params)

-- Define our network
function predict(params, input, target)
   local h1 = pool1(acts1(conv1(params.conv1, reshape(input))))
   local h2 = pool2(acts2(conv2(params.conv2, h1)))
   local h3 = linear(params.linear, flatten(h2))
   local out = util.logSoftMax(h3)
   return out
end

-- Define our loss function
function f(params, input, target)
   local prediction = predict(params, input, target)
   local loss = lossFuns.logMultinomialLoss(prediction, target)
   return loss, prediction
end

-- Get the gradients closure magically:
local df = grad(f, {
   optimize = true,              -- Generate fast code
   stableGradients = true,       -- Keep the gradient tensors stable so we can use CUDA IPC
})

return {
  params = params,
  f = f,
  df = df
}



-- layer 1:
-- conv[1], params[1] = grad.nn.SpatialConvolutionMM(3, 64, 5,5, 1,1, 2,2)
-- bn[1], params[2] = grad.nn.SpatialBatchNormalization(64, 1e-3)
-- acts[1] = grad.nn.ReLU()
-- pool[1] = grad.nn.SpatialMaxPooling(2,2, 2,2)

-- -- layer 2:
-- conv[2], params[3] = grad.nn.SpatialConvolutionMM(64, 128, 5,5, 1,1, 2,2)
-- bn[2], params[4] = grad.nn.SpatialBatchNormalization(128, 1e-3)
-- acts[2] = grad.nn.ReLU()
-- pool[2] = grad.nn.SpatialMaxPooling(2,2, 2,2)

-- -- layer 3:
-- conv[3], params[5] = grad.nn.SpatialConvolutionMM(128, 256, 5,5, 1,1, 2,2)
-- bn[3], params[6] = grad.nn.SpatialBatchNormalization(256, 1e-3)
-- acts[3] = grad.nn.ReLU()
-- pool[3] = grad.nn.SpatialMaxPooling(2,2, 2,2)

-- -- layer 4:
-- conv[4], params[7] = grad.nn.SpatialConvolutionMM(256, 512, 5,5, 1,1, 2,2)
-- bn[4], params[8] = grad.nn.SpatialBatchNormalization(512, 1e-3)
-- acts[4] = grad.nn.ReLU()
-- pool[4] = grad.nn.SpatialMaxPooling(2,2, 2,2)

-- -- layer 5:
-- flatten = grad.nn.Reshape(512*2*2)
-- linear,params[9] = grad.nn.Linear(512*2*2, 10)


-- Make sure all the nodes have the same parameter values

-- -- Loss:
-- local logSoftMax = grad.nn.LogSoftMax()
-- local crossEntropy = grad.nn.ClassNLLCriterion()


-- -- Define our network
-- local function predict(params, input, target)
--    local h = input
--    local np = 1
--    for i in ipairs(conv) do
--       h = pool[i](acts[i](bn[i](params[np+1], conv[i](params[np], h))))
--       np = np + 2
--    end
--    local hl = linear(params[np], flatten(h), 0.5)
--    local out = logSoftMax(hl)
--    return out
-- end

-- -- Define our loss function
-- local function f(params, input, target)
--    local prediction = predict(params, input, target)
--    local loss = crossEntropy(prediction, target)
--    return loss, prediction
-- end

-- -- Get the gradients closure magically:
-- local df = grad(f, {
--    optimize = true,              -- Generate fast code
--    stableGradients = true,       -- Keep the gradient tensors stable so we can use CUDA IPC
-- })