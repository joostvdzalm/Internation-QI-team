# Internation-QI-team - Group 4

Group members:

| Name                | email-address                       | Student number |
|---------------------|-------------------------------------|----------------|
| Boran Apak          | B.apak@student.tudelft.nl           | 4552342        |
| Yorgos Sotiropoulos | G.SOTIROPOULOS-1@student.tudelft.nl | 5231175        |
| Joost van der Zalm  | J.C.vanderzalm@student.tudelft.nl   | 4440943        |


# Proposal for the Fundamentals of Quantum Information Project

**Title:** Investigating Quantum process tomography on multiple backends

**Description:** 
Our goal is to implement quantum process tomography for 2 qubits. We would like to start from simple quantum state tomography for 1 and 2 qubits. Then we shall proceed to deal with quantum process tomography for 1 qubit and figure out what the results are for simple unitary rotations and generalise into single qubit processes. After fully understanding the single qubit case, we aim to tackle the 2 qubit case with the final aim of comprehending how the 240 different parameters describing such a process can be interpreted. Slowly building up in complexity so that we can keep a good understanding of what is happening. 

The end goal is that we delve into what actually is a quantum process and how it can be described by the chi-matrix mentioned in Nielsen & Chuang. We want to compare the differences between the Spin-2 and Starmon-5 qubits, as well as from simulations. We will try to quantify our results by measuring the time the algorithm takes on all platforms, but also the accuracy of the algorithm on different platforms. This will be done by looking at how well the algorithm could predict a certain predetermined set of operations, for example by measuring the variance. Later on we could look with more detail into how many times we repeat a process to apply tomography successfully. More towards the end of the project we could aim to optimize the tomography algorithm and see how many runs are necessary and what accuracy we could obtain. 

**Platform:** 
We will work with the quantum inspire platforms from QuTech, where at the beginning we will mainly focus on writing code for the QX quantum simulator. After we are content with the results on that platform we also want to use the Spin-2 hardware backend. For us it would be nice to compare the results we get from the actual on chip calculations and see if we can explain any differences. We would also like to work with the Starmon-5 hardware: we expect the systems to have restrictions and it would be a nice opportunity for us to work with the hardware backends, learn what the restrictions are and see how it impacts the measurements. 

**Motivations:** 
Boran: I think this is the perfect opportunity to not only apply what we learned in the fundamentals of quantum information course, but also gain a deeper theoretical understanding of what we've learned, because by really applying something you always find out that things turn out to be slightly different than you thought they were. 

Yorgos: I am really excited at the possibility of learning more about this fascinating field through this project and eventually really grasp the elusive concepts of quantum processes and their correlations to density matrices. 

Joost: This would be a good opportunity for me to get more experience with the experimental part of QI. Which is also nice to prepare me for my MSc thesis later on. Looking forward to the moment we can press run and the code finally works and gives the results we were aiming for!
