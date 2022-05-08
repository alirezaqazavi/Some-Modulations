function PHASE_CYLINDER(x,y,z,L)
%%  PHASE_CYLENDER (x,y,z,L) plots PHASE_CYLENDER
% x , y , z must be vectors with same size
% L is a parameter for LRC
%%
% create the graphics object
 h = plot3(NaN,NaN,NaN);
 % label the axes
 xlabel('I');
 ylabel('Q'); 
 zlabel('time(sec)');
%  title('phase cylinder');
% set limits
 xlim([min(x) max(x)]);
 ylim([min(y) max(y)]);
 zlim([min(z) max(z)]);
 % plot the data
 for k=1:length(x)
    set(h,'XData',x(1:k),'YData',y(1:k),'ZData',z(1:k));
    pause(0.0001);
 end
 grid on;legend(sprintf('L = %d',L));
end