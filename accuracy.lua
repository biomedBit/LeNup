-- Adapted from https://github.com/hpenedones/metrics

ROC = {}
ROC.__index = ROC

function variance_explained(Yf, preds)
    local num_seqs = Yf:dataspaceSize()[1]
    local Ydim = Yf:dataspaceSize()[2]

    local R2s = torch.Tensor(Ydim)
    for yi = 1,Ydim do
        -- read Yi from file
        local Yi = Yf:partial({1,num_seqs},{yi,yi}):squeeze()
        Yi = Yi:float()

        -- subset to preds i
        local preds_i = preds[{{},yi}]

        -- compute variances
        local Yi_var = (Yi - Yi:mean()):var()
        local preds_var = (Yi - preds_i):var()

        -- save
        R2s[yi] = 1.0 - preds_var/Yi_var
    end

    return R2s
end

function spearmanr(Yf, preds)
    local num_seqs = Yf:dataspaceSize()[1]
    local Ydim = Yf:dataspaceSize()[2]

    local n_denom = num_seqs*(num_seqs^2 - 1)
    local cors = torch.Tensor(Ydim)

    for yi = 1,Ydim do
        -- read Yi from file
        local Yi = Yf:partial({1,num_seqs},{yi,yi}):squeeze()

        -- subset to preds i
        local preds_i = preds[{{},yi}]

        -- argsort
        local _, Y_si = Yi:sort()
        local _, preds_si = preds_i:sort()

        -- assign ranks
        local Y_ranks = Y_si:clone()
        local preds_ranks = preds_si:clone()
        for ri = 1,num_seqs do
            Y_ranks[Y_si[ri]] = ri
            preds_ranks[preds_si[ri]] = ri
        end

        local d2_sum = (Y_ranks - preds_ranks):float():pow(2):sum()

        -- save
        cors[yi] = 1.0 - 6.0*d2_sum/n_denom
    end

    return cors
end


local function determine_roc_points_needed(responses_sorted, labels_sorted)
   	local npoints = 1
   	local i = 1
   	local nsamples = responses_sorted:size()[1]

   	while i<nsamples do
		local split = responses_sorted[i]
		while i <= nsamples and responses_sorted[i] == split do
			i = i+1
		end
		while i <= nsamples and labels_sorted[i] == 0 do
			i = i+1
		end
		npoints = npoints + 1
   	end
   	return npoints + 1
end


function ROC.points(responses, labels)

	-- assertions about the data format expected
	assert(responses:size():size() == 1, "responses should be a 1D vector")
	assert(labels:size():size() == 1 , "labels should be a 1D vector")

   	-- assuming labels {0, 1}
   	local npositives = torch.sum(torch.eq(labels, 1))
   	local nnegatives = torch.sum(torch.eq(labels, 0))
   	local nsamples = npositives + nnegatives

   	assert(nsamples == responses:size()[1], "labels should contain only 0 or 1 values")

   	-- sort by response value
   	local responses_sorted, indexes_sorted = torch.sort(responses)
   	local labels_sorted = labels:index(1, indexes_sorted)


  	local roc_num_points = determine_roc_points_needed(responses_sorted, labels_sorted)
   	local roc_points = torch.Tensor(roc_num_points, 2)

   	roc_points[roc_num_points][1], roc_points[roc_num_points][2]  = 1.0, 1.0

   	local npoints = 1
	local true_negatives = 0
	local false_negatives = 0
   	local i = 1

   	while i<nsamples do
		local split = responses_sorted[i]
		-- if samples have exactly the same response, can't distinguish
		-- between them with a threshold in the middle
		while i <= nsamples and responses_sorted[i] == split do
			if labels_sorted[i] == 0 then
				true_negatives = true_negatives + 1
			else
				false_negatives = false_negatives + 1
			end
			i = i+1
		end
		while i <= nsamples and labels_sorted[i] == 0 do
			true_negatives = true_negatives + 1
			i = i+1
		end
		npoints = npoints + 1
		local false_positives = nnegatives - true_negatives
		local true_positives = npositives - false_negatives
		local false_positive_rate = 1.0*false_positives/nnegatives
		local true_positive_rate = 1.0*true_positives/npositives
		roc_points[roc_num_points - npoints + 1][1] = false_positive_rate
		roc_points[roc_num_points - npoints + 1][2] = true_positive_rate
   	end

   	assert(roc_num_points - npoints + 1 == 2, "missed pre-allocated table target")

   	roc_points[1][1], roc_points[1][2] = 0.0, 0.0

   	return roc_points
end


function ROC.area(roc_points)

	local area = 0.0
	local npoints = roc_points:size()[1]

	for i=1,npoints-1 do
		local width = (roc_points[i+1][1] - roc_points[i][1])
		local avg_height = (roc_points[i][2]+roc_points[i+1][2])/2.0
		area = area + width*avg_height
	end

	return area
end