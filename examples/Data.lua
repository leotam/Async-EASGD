local Dataset = require 'dataset.Dataset'

-- Load the CIFAR-10 dataset

local function Data(partitionNum, partitions, batchSize, cuda)

  local lfs = require 'lfs'
  dirPrefix = string.match(lfs.currentdir(),"/%a+/%a+/")

  local trainingDataset = Dataset(dirPrefix .. 'Datasets/mnist/train.t7', {
     -- Partition dataset so each node sees a subset:
     partition = partitionNum,
     partitions = partitions
  })

  local testDataset = Dataset(dirPrefix .. 'Datasets/mnist/test.t7', {
     -- Partition dataset so each node sees a subset:
     partition = partitionNum,
     partitions = partitions
  })

  local getTrainingBatch, numTrainingBatches = trainingDataset.sampledBatcher({
     samplerKind = 'permutation',
     batchSize = batchSize,
     inputDims = { 1024 },
     verbose = true,
     cuda = cuda,
     processor = function(res, processorOpt, input)
        -- This function is not a closure, it is run in a clean Lua environment
        local image = require 'image'
        -- Turn the res string into a ByteTensor (containing the PNG file's contents)
        -- local bytes = torch.ByteTensor(#res)
        -- bytes:storage():string(res)
        -- Decompress the PNG bytes into a Tensor
        -- local pixels = image.decompressPNG(bytes)
        -- Copy the pixels tensor into the mini-batch
        input:copy(res:view(1, 1024))
        return true
     end,
  })


  local getTestBatch, numTestBatches = testDataset.sampledBatcher({
     samplerKind = 'permutation',
     batchSize = batchSize,
     inputDims = { 1024 },
     verbose = true,
     cuda = cuda,
     processor = function(res, processorOpt, input)
        -- This function is not a closure, it is run in a clean Lua environment
        local image = require 'image'
        -- Turn the res string into a ByteTensor (containing the PNG file's contents)
        -- local bytes = torch.ByteTensor(#res)
        -- bytes:storage():string(res)
        -- Decompress the PNG bytes into a Tensor
        -- local pixels = image.decompressPNG(bytes)
        -- Copy the pixels tensor into the mini-batch
        input:copy(res:view(1, 1024))
        return true
     end,
  })

  -- Load in MNIST
  local classes = {'0','1','2','3','4','5','6','7','8','9'}
  -- local confusionMatrix = optim.ConfusionMatrix(classes)

  return {
    getTrainingBatch = getTrainingBatch,
    numTrainingBatches = numTrainingBatches,
    getTestBatch = getTestBatch,
    numTestBatches = numTestBatches,
    classes = classes
  }
end

return Data
