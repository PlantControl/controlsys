function generate_matlab_least_squares(output)
if nargin == 0
    output = fullfile(fileparts(mfilename('fullpath')), 'matlab-least-squares.json');
end
models = {
    struct('name','second-order','num',[1 .7],'den',[1 2.5 4],'sampleTime',.2,'orders',[0 1 2 4]),
    struct('name','fast-dynamics','num',1,'den',[1 21 20],'sampleTime',.4,'orders',[1 2]),
    struct('name','integrator','num',1,'den',[1 1 0],'sampleTime',.1,'orders',2)
};
cases = {};
for index = 1:numel(models)
    model = models{index};
    source = tf(model.num, model.den);
    for fitOrder = model.orders
        options = c2dOptions('Method','least-squares');
        if fitOrder > 0
            options.FitOrder = fitOrder;
        end
        converted = c2d(source, model.sampleTime, options);
        [num,den] = tfdata(tf(converted),'v');
        num = num / den(1);
        den = den / den(1);
        frequencies = (pi/model.sampleTime)*[.0001 .007 .04 .13 .37 .61 .89 .999];
        response = reshape(freqresp(converted, frequencies),1,[]);
        cases{end+1} = struct('name',sprintf('%s-order-%d',model.name,fitOrder), ...
            'sourceNumerator',model.num,'sourceDenominator',model.den, ...
            'sampleTime',model.sampleTime,'fitOrder',fitOrder, ...
            'numerator',num,'denominator',den,'frequencies',frequencies, ...
            'responseReal',real(response),'responseImag',imag(response));
    end
end
controlVersion = ver('control');
reference = struct('matlabVersion',version,'controlVersion',controlVersion.Version, ...
    'coefficientTolerance',1e-6,'responseTolerance',1e-6,'cases',{cases});
file = fopen(output,'w');
assert(file >= 0, 'Cannot open fixture output');
cleanup = onCleanup(@() fclose(file));
fprintf(file,'%s\n',jsonencode(reference,'PrettyPrint',true));
end
