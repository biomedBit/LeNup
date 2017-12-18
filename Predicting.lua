#!/usr/bin/env th

require 'hdf5'


----------------------------------------------------------------
-- parse arguments
----------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('DNA ConvNet testing')
cmd:text()
cmd:text('Arguments')
cmd:argument('model_file')
cmd:argument('data_file')
cmd:argument('out_file')
cmd:text()
cmd:text('Options:')
cmd:option('-batch', 128, 'Batch size')
cmd:option('-cuda', false, 'Run on GPGPU')
cmd:option('-cudnn', false, 'Run on GPGPU w/ cuDNN')
cmd:text()
opt = cmd:parse(arg)

-- set cpu/gpu
cuda_nn = opt.cudnn
cuda = opt.cuda or opt.cudnn
require 'convnet'

----------------------------------------------------------------
-- load data
----------------------------------------------------------------
local data_open = hdf5.open(opt.data_file, 'r')
local test_seqs = data_open:read('test_in')

----------------------------------------------------------------
-- construct model
----------------------------------------------------------------
-- initialize
local convnet = ConvNet:__init()

-- load from saved parameters
local convnet_params = torch.load(opt.model_file)
convnet:load(convnet_params)

if cuda then
    convnet:cuda()
end

----------------------------------------------------------------
-- predict and test
----------------------------------------------------------------
-- guarantee evaluate mode
convnet.model:evaluate()

-- measure accuracy on a test set
local preds, scores = convnet:predict(test_seqs, opt.batch)

-- close HDF5
data_open:close()


----------------------------------------------------------------
-- dump to file
----------------------------------------------------------------
local values = preds

local predict_out = io.open(opt.out_file, 'w')

-- print predictions
for si=1,(#values)[1] do
    predict_out:write(values[{si,1}])
    for ti=2,(#values)[2] do
        predict_out:write(string.format("\t%s",values[{si,ti}]))
    end
    predict_out:write("\n")
end

predict_out:close()
