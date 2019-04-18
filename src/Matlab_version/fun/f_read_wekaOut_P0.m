function [pred, actural_label] = f_read_wekaOut_P0(fn)
% read the predicted probability from weka output 
f = fopen(fn, 'r');
%pred = zeros(4000,1);
pred=[];
actural_label = [];

for i = 1:5
   fgetl(f);
end

while ~feof(f)  
   line1 = fgetl(f);
   
   if length(line1) < 36
       break;
   end
   
   line = line1(36:end);
   
   sco = sscanf(line, '%f');
   if strcmp(line1(28), '0');
       sco = 1 - sco;
       %fprintf(line1);
       %fprintf('\n');
   end
   %line, sco,
   
   pred = [pred; sco];
   actural_label = [actural_label; sscanf(line1(17), '%f')];
   % i=i+1;
end

fclose(f);


end

