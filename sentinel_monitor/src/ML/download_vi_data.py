import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from services.sentinel_service import SentinelService

def main():
    sentinel_service = SentinelService()

    # Exemplo de uso com coordenadas de uma área de interesse
    geometry = {
        "type": "Polygon",
        "coordinates": [
            [
                [-5.0, 40.0],
                [-5.0, 45.0],
                [5.0, 45.0],
                [5.0, 40.0],
                [-5.0, 40.0]
            ]
        ]
    }
    
    # Baixar imagens dos índices de vegetação
    sentinel_service.download_vi_image(geometry, "2023-01-01", "2023-01-31")
    
    # Baixar imagens de biomassa
    sentinel_service.download_biomass_image(geometry, "2023-01-01", "2023-01-31")

if __name__ == "__main__":
    main()
