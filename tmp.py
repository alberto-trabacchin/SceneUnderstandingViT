def train_mpl(
    X_unl: np.ndarray,
    X_lab: np.ndarray,
    Y_lab: np.ndarray,
    num_epochs: int = 10,
    batch_size: int = 16,
    log_every: int = 10,
    train_seed: int | None = None,
) -> Classifier:

    torch.autograd.set_detect_anomaly(False)

    if train_seed is not None:
        np.random.seed(train_seed)
        torch.manual_seed(train_seed)

    teacher = Model().cuda()
    student = Model().cuda()

    with torch.no_grad():

        for p in student.parameters():
            p.uniform_(-1., 1.)

        for p in teacher.parameters():
            p.uniform_(-1., 1.)

    teacher_optim = torch.optim.SGD(params=teacher.parameters(), lr=0.1)
    student_optim = torch.optim.SGD(params=student.parameters(), lr=0.1)

    all_idxs_lab = np.arange(X_lab.shape[0])
    all_idxs_unl = np.arange(X_unl.shape[0])

    step = 0
    for epoch in range (num_epochs):
        np.random.shuffle(all_idxs_lab)
        np.random.shuffle(all_idxs_unl)

        # shuffle labeled examples
        x_epoch_lab = X_lab[all_idxs_lab, :].reshape(-1, batch_size, 2)
        y_epoch_lab = Y_lab[all_idxs_lab].reshape(-1, batch_size)

        # shuffle unlabeled examples
        x_epoch_unl = X_unl[all_idxs_unl, :].reshape(-1, batch_size * 8, 2)

        for i in range(x_epoch_lab.shape[0]):
            x_lab = x_epoch_lab[i]
            y_lab = torch.tensor(y_epoch_lab[i], dtype=torch.int64, device="cuda")

            x_unl = x_epoch_unl[i]

            teacher_optim.zero_grad()

            teacher_lab_logits = teacher.forward(x_lab)
            teacher_unl_logits = teacher.forward(x_unl)
            teacher_unl_labels = F.softmax(teacher_unl_logits, dim=1)

            student_lab_logits = student.forward(x_lab)
            student_unl_logits = student.forward(x_unl)

            # train the student
            student_optim.zero_grad()
            student_loss = F.cross_entropy(student_lab_logits, y_lab, reduction="mean")
            student_loss.backward()
            student_grad_1 = [p.grad.data.clone().detach() for p in student.parameters()]

            # train the student
            student_optim.zero_grad()
            student_loss = F.cross_entropy(student_unl_logits, teacher_unl_labels.detach(), reduction="mean")
            student_loss.backward()
            student_grad_2 = [p.grad.data.clone().detach() for p in student.parameters()]
            student_optim.step()

            mpl_coeff = sum([torch.dot(g_1.ravel(), g_2.ravel()).sum().detach().item() for g_1, g_2 in zip(student_grad_1, student_grad_2)])

            # train the teacher
            teacher_optim.zero_grad()
            teacher_loss_ent = F.cross_entropy(teacher_lab_logits, y_lab, reduction="mean")
            teacher_loss_mpl = mpl_coeff * F.cross_entropy(teacher_unl_logits, teacher_unl_labels, reduction="mean")

            teacher_loss = teacher_loss_ent + teacher_loss_mpl
            teacher_loss.backward()
            teacher_optim.step()

            step += 1
            if step % log_every == 0 or (i == x_epoch_lab.shape[0] - 1 and epoch == num_epochs - 1):
                print(f"step={step:<7d}"
                      f"teacher_loss={teacher_loss.item():<7.4f}"
                      f"student_loss={student_loss.item():<7.4f}",
                      flush=True)

    return student

classifier_mpl = train_mpl(
    x_all,
    x_lab,
    y_lab,
    num_epochs=20_000,
    batch_size=y_lab.size,
    log_every=1_000,
    train_seed=0)

plot_moons(x_all, y_all, x_lab, y_lab, fn=classifier_mpl)