from fgir_kd.model_utils.build_model import build_model
from fgir_kd.other_utils.build_args import parse_train_args
from fgir_kd.data_utils.build_dataloaders import build_dataloaders


def main():
    args = parse_train_args()
    train_loader, val_loader, test_loader = build_dataloaders(args)

    model = build_model(args)
    print('Original model: ')
    for name, layer in model.named_modules():
        print(name)

    if args.model_name_teacher:
        model_t = build_model(args, teacher=True)
        print('Teacher model: ')
        for name, layer in model_t.named_modules():
            print(name)

        if args.cont_loss or (args.kd_aux_loss == 'crd' and args.selector == 'cal'):
            args.if_channels = model_t.model.if_channels
        elif args.kd_aux_loss == 'crd':
            args.if_channels = model_t.if_channels

        model_s = build_model(args, student=True)
        print('Student model: ')
        for name, layer in model_s.named_modules():
            print(name)

    return 0


if __name__ == '__main__':
    main()

