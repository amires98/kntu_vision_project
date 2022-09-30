# kntu_vision_project
The final project of the course "Fundamentals of Computer Vision" K. N. Toosi University of Technology by Dr. Behrooz Nasihatkon
The objective of this project is to producing BEV (bird’s eye view) perspective of a soccer match live stream.

My solution consists of tree major parts. First,
extracting features to compute the desired homography matrix between
the two perspectives. Second, extracting players and referee
(patches) using image processing methods and compute their position
in the BEV perspective using the computed homography matrix.
Third, train a CNN capable of classifying extracted patches into
three class, team A, team B and the referee. So the input is a soccer
match stream of any perspective and the output is a stream of the
field from BEV view with colored circles representing players and
the referee. (PyTorch, OpenCV)
