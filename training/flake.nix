{
  description = "uv-managed Python template (impure build), with runtime GPU shim + nix-ld container";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs =
    { self, nixpkgs }:

    let
      lib = nixpkgs.lib;

      systems = [
        "x86_64-linux"
        "aarch64-darwin"
      ];

      forAllSystems = f: lib.genAttrs systems (system: f system);

      pythonSpec = "3.14.2";

      appName = "train";
      entrypoint = "train";

      # NixOS GPU shim
      gpuLibPath = "/run/opengl-driver/lib:/run/opengl-driver-32/lib";

    in
    {

      devShells = forAllSystems (
        system:
        let
          pkgs = import nixpkgs { inherit system; };

          toolchain = [
            pkgs.uv
            pkgs.ruff
            pkgs.cacert
            pkgs.makeWrapper
            pkgs.ty
            pkgs.zlib
            pkgs.openssl
            pkgs.stdenv.cc
          ];

          isLinux = pkgs.stdenv.isLinux;

        in
        {
          default = pkgs.mkShell {
            packages = toolchain;

            env = {
              UV_MANAGED_PYTHON = "1";
              UV_PROJECT_ENVIRONMENT = ".venv";
            };

            shellHook = ''
              set -euo pipefail


              ${lib.optionalString isLinux ''
                # GPU shim for local dev on NixOS
                if [ -d /run/opengl-driver/lib ]; then
                  export LD_LIBRARY_PATH="${gpuLibPath}:''${LD_LIBRARY_PATH:-}"
                fi
              ''}
            '';
          };
        }
      );

      packages = forAllSystems (
        system:
        let
          pkgs = import nixpkgs { inherit system; };

          isLinux = pkgs.stdenv.isLinux;

          toolchain = [
            pkgs.uv
            pkgs.ruff
            pkgs.cacert
            pkgs.makeWrapper
            pkgs.ty
            pkgs.zlib
            pkgs.openssl
            pkgs.stdenv.cc
          ];

          uvBundle = pkgs.stdenvNoCC.mkDerivation {
            pname = "${appName}-uv-bundle";
            version = "0.1.0";
            src = ./.;

            # Requires relaxed sandbox / network
            __noChroot = true;
            allowSubstitutes = true;
            dontFixup = true;

            nativeBuildInputs = toolchain;

            installPhase = ''
              set -euo pipefail


              export HOME="$TMPDIR/home"
              mkdir -p "$HOME"

              export UV_CACHE_DIR="$TMPDIR/uv-cache"
              export UV_MANAGED_PYTHON=1

              export UV_PYTHON_INSTALL_DIR="$out/python"
              export UV_PROJECT_ENVIRONMENT="$out/venv"

              uv python install ${pythonSpec}
              uv venv --python ${pythonSpec}
              uv sync --frozen --no-dev --no-editable
            '';
          };

          cli = pkgs.stdenvNoCC.mkDerivation {
            pname = appName;
            version = "0.1.0";

            dontUnpack = true;
            nativeBuildInputs = [ pkgs.makeWrapper ];

            installPhase = ''
              set -euo pipefail
              mkdir -p "$out/bin"


              makeWrapper "${uvBundle}/venv/bin/${entrypoint}" "$out/bin/${entrypoint}" \
                --run 'if [ -d /run/opengl-driver/lib ]; then export LD_LIBRARY_PATH="${gpuLibPath}:''${LD_LIBRARY_PATH:-}"; fi'
              makeWrapper "${uvBundle}/venv/bin/train-latent" "$out/bin/train-latent" \
                --run 'if [ -d /run/opengl-driver/lib ]; then export LD_LIBRARY_PATH="${gpuLibPath}:''${LD_LIBRARY_PATH:-}"; fi'
              makeWrapper "${uvBundle}/venv/bin/train-vae" "$out/bin/train-vae" \
                --run 'if [ -d /run/opengl-driver/lib ]; then export LD_LIBRARY_PATH="${gpuLibPath}:''${LD_LIBRARY_PATH:-}"; fi'
              makeWrapper "${uvBundle}/venv/bin/train-legacy-ddpm" "$out/bin/train-legacy-ddpm" \
                --run 'if [ -d /run/opengl-driver/lib ]; then export LD_LIBRARY_PATH="${gpuLibPath}:''${LD_LIBRARY_PATH:-}"; fi'
              makeWrapper "${uvBundle}/venv/bin/sample-latent" "$out/bin/sample-latent" \
                --run 'if [ -d /run/opengl-driver/lib ]; then export LD_LIBRARY_PATH="${gpuLibPath}:''${LD_LIBRARY_PATH:-}"; fi'
              makeWrapper "${uvBundle}/venv/bin/torchrun" "$out/bin/torchrun" \
                --run 'if [ -d /run/opengl-driver/lib ]; then export LD_LIBRARY_PATH="${gpuLibPath}:''${LD_LIBRARY_PATH:-}"; fi'
              makeWrapper "${uvBundle}/venv/bin/accelerate" "$out/bin/accelerate" \
                --run 'if [ -d /run/opengl-driver/lib ]; then export LD_LIBRARY_PATH="${gpuLibPath}:''${LD_LIBRARY_PATH:-}"; fi'
              makeWrapper "${uvBundle}/venv/bin/python" "$out/bin/python" \
                --run 'if [ -d /run/opengl-driver/lib ]; then export LD_LIBRARY_PATH="${gpuLibPath}:''${LD_LIBRARY_PATH:-}"; fi'

            '';
          };

          libDir = "/lib64";

          nixLdSetup = pkgs.runCommand "nix-ld-setup" { } ''
            set -euo pipefail
            mkdir -p "$out${libDir}"
            install -D -m755 ${pkgs.nix-ld}/libexec/nix-ld \
              "$out${libDir}/$(basename ${pkgs.stdenv.cc.bintools.dynamicLinker})"
          '';

          image = pkgs.dockerTools.buildImage {
            name = appName;
            tag = "latest";

            copyToRoot = pkgs.buildEnv {
              name = "image-root";

              paths = [
                cli
                uvBundle

                pkgs.coreutils
                pkgs.bash
                pkgs.git
                pkgs.findutils
                pkgs.cacert

                # nix-ld shim at /lib64/...
                nixLdSetup
              ];

              pathsToLink = [
                "/bin"
                "/lib"
                "/lib64"
                "/usr"
              ];
            };

            config = {
              Env = [
                "NIX_LD=${pkgs.stdenv.cc.bintools.dynamicLinker}"
                "NIX_LD_LIBRARY_PATH=${
                  pkgs.lib.makeLibraryPath [
                    pkgs.stdenv.cc.cc.lib
                    pkgs.glibc
                    pkgs.zlib
                    pkgs.openssl
                  ]
                }"

                # NVIDIA runtime typically injects libs here:
                "LD_LIBRARY_PATH=/usr/lib64"
              ];

              Cmd = [ "${cli}/bin/${entrypoint}" ];
            };

            runAsRoot = ''
              #!${pkgs.runtimeShell}
              set -euo pipefail
              mkdir -p /etc/ssl/certs
              ln -sf ${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt /etc/ssl/certs/ca-certificates.crt
              ln -sf ${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt /etc/ssl/certs/ca-bundle.crt
            '';
          };

        in
        {
          uv-bundle = uvBundle;
          ${appName} = cli;

          default = cli;
        }
        // lib.optionalAttrs isLinux {
          image = image;
        }
      );

      apps = forAllSystems (
        system:
        let
          cli = self.packages.${system}.${appName};
        in
        {
          ${appName} = {
            type = "app";
            program = "${cli}/bin/${entrypoint}";
          };

          default = {
            type = "app";
            program = "${cli}/bin/${entrypoint}";
          };
        }
      );

    };
}
