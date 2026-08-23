classdef PositionalEncodingLayer < nnet.layer.Layer & nnet.layer.Formattable
    % Custom Positional Encoding Layer for the HA-PIT Transformer
    
    properties
        NumChannels % The hidden dimension size (e.g., 32)
        ScaleFactor % The scalar to prevent drowning out the CNN data
    end
    
    methods
        function layer = PositionalEncodingLayer(numChannels, scale, layerName)
            % The constructor name MUST match the classdef name exactly!
            layer.Name = layerName;
            layer.Description = "Positional Encoding (Clock)";
            layer.NumChannels = numChannels;
            layer.ScaleFactor = scale;
        end
        
        function Z = predict(layer, X)
            % X format is typically 'CBT' (Channels, Batch, Time)
            seqLen = size(X, 3);
            
            % Initialize the 2D encoding matrix matching the data type of X
            pos_enc_2d = zeros(layer.NumChannels, seqLen, 'like', X);
            
            % Generate the Sine and Cosine waves dynamically
            for pos = 1:seqLen
                for i = 1:2:layer.NumChannels
                    exponent = (i-1) / layer.NumChannels;
                    pos_enc_2d(i, pos)   = sin((pos-1) / (10000^exponent));
                    if i+1 <= layer.NumChannels
                        pos_enc_2d(i+1, pos) = cos((pos-1) / (10000^exponent));
                    end
                end
            end
            
            % Scale the clock signal
            pos_enc_2d = pos_enc_2d * layer.ScaleFactor;
            
            % Reshape into [Channels, 1, Time] for batch broadcasting
            pos_enc_3d = reshape(pos_enc_2d, [layer.NumChannels, 1, seqLen]);
            
            % Add the clock signal directly to the incoming CNN data
            Z = X + pos_enc_3d; 
        end
    end
end