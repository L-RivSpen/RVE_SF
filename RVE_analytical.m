%%% Parameters 
nu = 0.3;
mu = 0.5 * 10^6;
k = (2* mu *(1 + nu)) / (3 * (1 - 2* nu));
c_1 = mu / 2;

%%% Defining F
F_avg = [1.00975 0.0001695;
         0.000146 1.00015];

%%% SEF
syms F11 F12 F21 F22

F = [F11 F12;
     F21 F22];

F_inv = inv(F);
F_invT = F_inv.';

J = det(F);
B = F * F.';
sef_vol = 0.25*k*(J^2 - 1- 2*log(J));
sef_iso = c_1 *(trace(B) -2);

sef = sef_vol + sef_iso;

%%% P 
P11 = diff(sef,F11);
P12 = diff(sef,F12);
P21 = diff(sef,F21);
P22 = diff(sef,F22);

P_AD = [P11 P12;
        P21 P22];

P_anal = 0.5*k*(J^2 -1)*F_invT + 2*c_1*F;

P_AD_num = subs(P_AD, F,F_avg)
P_anal_num = subs(P_anal, F, F_avg)

P_returned = [513781 2.84;
              126    512323]

Perc_diff = ((P_AD_num - P_returned)/P_returned) * 100