function pred = f_read_wekaOut_P0_2(fn)
% read the predicted CLASS LABEL from weka output 

f = fopen(fn, 'r');
%pred = zeros(4000,1);
pred=[];
for i = 1:5
   fgetl(f);
end

while ~feof(f)  
   line1 = fgetl(f);
   
   if length(line1) < 36
       break;
   end
   
   line = line1(28);
   
   sco = logical(sscanf(line, '%d'));

   pred = [pred; sco];
    
   % i=i+1;
end

fclose(f);


end

