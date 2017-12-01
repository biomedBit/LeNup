-- Part of this code is adapted from https://github.com/davek44/Basset
Batcher = {}
Batcher.__index = Batcher

function Batcher:__init(Xf, Yf, batch_size)
    bat = {}
    setmetatable(bat, self)

    bat.Xf = Xf
    bat.num_seqs = Xf:dataspaceSize()[1]
    bat.init_depth = Xf:dataspaceSize()[2]
    bat.seq_len = Xf:dataspaceSize()[4]

    bat.Yf = Yf
    if bat.Yf ~= nil then
        bat.num_targets = Yf:dataspaceSize()[2]
    end

    bat.batch_size = batch_size

    bat:reset()

    return bat
end

function Batcher:next()
    local X_batch = nil
    local Y_batch = nil

    if self.start <= self.num_seqs then
        -- get data
        X_batch = self.Xf:partial({self.start,self.stop},  {1,self.init_depth}, {1,1},{1,self.seq_len}):double()
        if self.Yf ~= nil then
            Y_batch = self.Yf:partial({self.start,self.stop}, {1,self.num_targets}):double()
        end

        -- update batch indexes for next
        self.start = self.start + self.batch_size
        self.stop = math.min(self.stop + self.batch_size, self.num_seqs)
    end
	
    return X_batch, Y_batch
end

function Batcher:reset()
    self.start = 1
    self.stop = math.min(self.start+self.batch_size-1, self.num_seqs)
end
